//! Long-term (pitch) prediction on block-mode residuals (format v17, research).
//!
//! After LPC/fixed (and cross-channel, and stage-2) prediction, a subframe's coded residual can
//! still repeat at a lag longer than LPC's order-32 reach: a pitch period, or the period of a
//! harmonic. This stage predicts each residual from K residuals around one lag T of the same
//! subframe, as MPEG-4 ALS's LTP and SRLA's pitch predictor do, and codes what is left:
//!
//! ```text
//! coded[n] = e[n] - ((sum_j g[j] * e[n - T + j - K/2] + 16) >> 5)   for n >= T + K/2
//! coded[n] = e[n]                                                  for n <  T + K/2
//! ```
//!
//! with T in 32..=2047, K in `TAP_COUNTS`, g[j] 7-bit signed integers (steps of 1/32), `>>` an
//! arithmetic shift. Only the subframe's own residual is used, so frames stay independent. The
//! encoder enables it per subframe only where the subframe gets smaller (1 flag bit otherwise).
//!
//! Range rule: in a subframe that uses this stage every residual e[n]
//! (before coding, i.e. what the decoder reconstructs) lies within +-`LIMIT` = 2^20. Then
//! |sum| <= 9 * 64 * 2^20 < 2^30 and the whole prediction is exact in 32-bit arithmetic, which is
//! what lets the decoder run it 8 lanes at a time. Since T - K/2 >= 28, every prediction in a run of
//! T - K/2 consecutive samples reads only samples before that run, so the inverse is computed one
//! such run at a time with no serial dependency inside it. A decoder rejects a stream whose
//! reconstruction leaves the range.
use crate::bitio::{BitReader, BitReaderError, BitWriter};
use crate::detmath;

pub const MIN_LAG: usize = 32;
const LAG_BITS: u32 = 11;
pub const MAX_LAG: usize = (1 << LAG_BITS) - 1;
/// Tap counts selectable per subframe (2-bit code), all odd so the taps centre on the lag.
pub const TAP_COUNTS: [usize; 4] = [1, 3, 5, 9];
const MAX_TAPS: usize = 9;
const SHIFT: u32 = 5;
const TAP_BITS: u32 = 7;
const TAP_MAX: i32 = (1 << (TAP_BITS - 1)) - 1;
/// Residual magnitude bound in a subframe that uses this stage (see the module docs).
pub const LIMIT: i64 = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Params { pub lag: usize, pub taps: Vec<i32> }

impl Params {
    /// Header: 1 bit enabled; then 11 bits lag (32..=2047), 2 bits tap-count code, 7 bits per tap.
    pub fn write(p: Option<&Params>, w: &mut BitWriter) {
        let Some(p) = p else { w.write_bits(0, 1); return };
        w.write_bits(1, 1);
        w.write_bits(p.lag as u64, LAG_BITS);
        w.write_bits(TAP_COUNTS.iter().position(|&k| k == p.taps.len()).expect("valid tap count") as u64, 2);
        for &g in &p.taps { w.write_signed(g as i64, TAP_BITS); }
    }
    pub fn read(r: &mut BitReader) -> Result<Option<Params>, String> {
        let e = |e: BitReaderError| e.0.to_string();
        if r.read_bits(1).map_err(e)? == 0 { return Ok(None); }
        let lag = r.read_bits(LAG_BITS).map_err(e)? as usize;
        if lag < MIN_LAG { return Err(format!("invalid long-term prediction lag {lag} (corrupted stream?)")); }
        let k = TAP_COUNTS[r.read_bits(2).map_err(e)? as usize];
        let mut taps = Vec::with_capacity(k);
        for _ in 0..k { taps.push(r.read_signed(TAP_BITS).map_err(e)? as i32); }
        Ok(Some(Params { lag, taps }))
    }
    /// Header bits including the flag.
    pub fn header_bits(&self) -> u64 { 1 + LAG_BITS as u64 + 2 + TAP_BITS as u64 * self.taps.len() as u64 }
    /// First position the prediction applies to.
    fn first(&self) -> usize { self.lag + self.taps.len() / 2 }
}

/// Encoder direction. `e` must be within +-`LIMIT`, which `search` checks before choosing this
/// stage; then the prediction is exact in i32 lanes (see the module docs), as `forward_reference`
/// (the definition, in i64) is tested to agree.
pub fn forward(p: &Params, e: &[i64]) -> Vec<i64> {
    match p.taps.len() {
        1 => forward_k::<1>(p, e),
        3 => forward_k::<3>(p, e),
        5 => forward_k::<5>(p, e),
        9 => forward_k::<9>(p, e),
        _ => unreachable!("tap count comes from TAP_COUNTS"),
    }
}

fn forward_k<const K: usize>(p: &Params, e: &[i64]) -> Vec<i64> {
    let g: [i32; K] = p.taps[..].try_into().expect("K taps");
    let x: Vec<i32> = e.iter().map(|&v| v as i32).collect();
    let back = p.lag + K / 2;
    let mut out = e.to_vec();
    let first = p.first().min(e.len());
    for (n, o) in out.iter_mut().enumerate().skip(first) {
        let w = &x[n - back..n - back + K];
        let mut a = 1i32 << (SHIFT - 1);
        for j in 0..K { a += g[j] * w[j]; }
        *o -= (a >> SHIFT) as i64;
    }
    out
}

/// The definition of the encoder direction, in 64-bit arithmetic (tests only).
pub fn forward_reference(p: &Params, e: &[i64]) -> Vec<i64> {
    let h = p.taps.len() / 2;
    let mut out = e.to_vec();
    for n in p.first()..e.len() {
        let s: i64 = p.taps.iter().enumerate().map(|(j, &g)| g as i64 * e[n + j - p.lag - h]).sum();
        out[n] = e[n] - ((s + (1 << (SHIFT - 1))) >> SHIFT);
    }
    out
}

/// Decoder direction, in place; errors if a reconstructed residual leaves +-`LIMIT`.
pub fn inverse(p: &Params, e: &mut [i64]) -> Result<(), &'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: AVX2 confirmed at runtime by `avx2_enabled`.
            return unsafe { inverse_avx2(p, e) };
        }
    }
    inverse_portable(p, e)
}

/// The same code compiled for the baseline target only (SSE2 on x86-64), for tests and timing comparisons.
pub fn inverse_portable(p: &Params, e: &mut [i64]) -> Result<(), &'static str> { inverse_impl(p, e) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inverse_avx2(p: &Params, e: &mut [i64]) -> Result<(), &'static str> {
    match p.taps.len() {
        1 => inverse_avx2_k::<1>(p, e),
        3 => inverse_avx2_k::<3>(p, e),
        5 => inverse_avx2_k::<5>(p, e),
        9 => inverse_avx2_k::<9>(p, e),
        _ => unreachable!("tap count comes from TAP_COUNTS"),
    }
}

/// `inverse_k` with explicit AVX2: the same 8-sample chunks, the prediction in i32 lanes
/// (`vpmulld`, exact under the range rule), the residual update and range check in i64 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn inverse_avx2_k<const K: usize>(p: &Params, e: &mut [i64]) -> Result<(), &'static str> {
    use std::arch::x86_64::*;
    let (m, h) = (e.len(), K / 2);
    let first = p.first().min(m);
    let back = p.lag + h;
    with_scratch(m, |x| {
    let mut bad = false;
    for (xo, &v) in x[..first].iter_mut().zip(&e[..first]) {
        bad |= v.unsigned_abs() > LIMIT as u64;
        *xo = v as i32;
    }
    let mut g = [_mm256_setzero_si256(); K];
    for j in 0..K { g[j] = _mm256_set1_epi32(p.taps[j]); }
    let round = _mm256_set1_epi32(1 << (SHIFT - 1));
    let (hi_lim, lo_lim) = (_mm256_set1_epi64x(LIMIT), _mm256_set1_epi64x(-LIMIT));
    let even = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    let mut badv = _mm256_setzero_si256();
    let xp = x.as_mut_ptr();
    let ep = e.as_mut_ptr();
    let mut n = first;
    while n + 8 <= m {
        let mut acc = round;
        let src = xp.add(n - back);
        for j in 0..K {
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(g[j], _mm256_loadu_si256(src.add(j) as *const __m256i)));
        }
        let pr = _mm256_srai_epi32::<{ SHIFT as i32 }>(acc);
        let r0 = _mm256_add_epi64(_mm256_loadu_si256(ep.add(n) as *const __m256i), _mm256_cvtepi32_epi64(_mm256_castsi256_si128(pr)));
        let r1 = _mm256_add_epi64(_mm256_loadu_si256(ep.add(n + 4) as *const __m256i), _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(pr)));
        badv = _mm256_or_si256(badv, _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpgt_epi64(r0, hi_lim), _mm256_cmpgt_epi64(lo_lim, r0)),
            _mm256_or_si256(_mm256_cmpgt_epi64(r1, hi_lim), _mm256_cmpgt_epi64(lo_lim, r1))));
        _mm256_storeu_si256(ep.add(n) as *mut __m256i, r0);
        _mm256_storeu_si256(ep.add(n + 4) as *mut __m256i, r1);
        let l0 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(r0, even));
        let l1 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(r1, even));
        _mm256_storeu_si256(xp.add(n) as *mut __m256i, _mm256_set_m128i(l1, l0));
        n += 8;
    }
    bad |= _mm256_testz_si256(badv, badv) == 0;
    for n in n..m {
        let mut a = 1i32 << (SHIFT - 1);
        for j in 0..K { a = a.wrapping_add(p.taps[j].wrapping_mul(x[n - back + j])); }
        let r = e[n].wrapping_add((a >> SHIFT) as i64);
        bad |= r.unsigned_abs() > LIMIT as u64;
        e[n] = r;
        x[n] = r as i32;
    }
    if bad { Err("long-term prediction residual out of range (corrupted stream?)") } else { Ok(()) }
    })
}

#[inline(always)]
fn inverse_impl(p: &Params, e: &mut [i64]) -> Result<(), &'static str> {
    match p.taps.len() {
        1 => inverse_k::<1>(p, e),
        3 => inverse_k::<3>(p, e),
        5 => inverse_k::<5>(p, e),
        9 => inverse_k::<9>(p, e),
        _ => unreachable!("tap count comes from TAP_COUNTS"),
    }
}

/// K taps as a constant, so each prediction is an unrolled multiply-add chain. Since every
/// prediction reads values at least `T - K/2 >= 28` positions back, any 8 consecutive samples are
/// independent of each other: they are computed together as one 8-lane i32 vector (exact under the
/// range rule), straight through the subframe, with a scalar tail of at most 7. The arithmetic is
/// explicitly wrapping: once a corrupted stream has produced a value past `LIMIT` (`bad`, reported
/// at the end) later sums may overflow `i32`; release builds already wrapped, this makes debug and
/// sanitizer builds do the same instead of panicking (found by the `chunk_payload` fuzz target).
#[inline(always)]
fn inverse_k<const K: usize>(p: &Params, e: &mut [i64]) -> Result<(), &'static str> {
    const ERR: &str = "long-term prediction residual out of range (corrupted stream?)";
    const L: usize = 8;
    let g: [i32; K] = p.taps[..].try_into().expect("K taps");
    let (m, h) = (e.len(), K / 2);
    let first = p.first().min(m);
    let back = p.lag + h; // x[n - back + j] is tap j's input for sample n
    with_scratch(m, |x| {
    let mut bad = false;
    for (xo, &v) in x[..first].iter_mut().zip(&e[..first]) {
        bad |= v.unsigned_abs() > LIMIT as u64;
        *xo = v as i32;
    }
    let mut n = first;
    while n + L <= m {
        let mut a = [1i32 << (SHIFT - 1); L];
        for j in 0..K {
            let s: [i32; L] = x[n - back + j..n - back + j + L].try_into().unwrap();
            for l in 0..L { a[l] = a[l].wrapping_add(g[j].wrapping_mul(s[l])); }
        }
        let ev: &mut [i64; L] = (&mut e[n..n + L]).try_into().unwrap();
        let mut xs = [0i32; L];
        for l in 0..L {
            let r = ev[l].wrapping_add((a[l] >> SHIFT) as i64);
            bad |= r.unsigned_abs() > LIMIT as u64;
            ev[l] = r;
            xs[l] = r as i32;
        }
        x[n..n + L].copy_from_slice(&xs);
        n += L;
    }
    for n in n..m {
        let mut a = 1i32 << (SHIFT - 1);
        for j in 0..K { a = a.wrapping_add(g[j].wrapping_mul(x[n - back + j])); }
        let r = e[n].wrapping_add((a >> SHIFT) as i64);
        bad |= r.unsigned_abs() > LIMIT as u64;
        e[n] = r;
        x[n] = r as i32;
    }
    if bad { Err(ERR) } else { Ok(()) }
    })
}

/// Runs `f` on a per-thread i32 buffer of at least `m` values, reused across subframes. Its
/// contents are stale on entry: both inverses write every position before reading it.
#[inline(always)]
fn with_scratch<R>(m: usize, f: impl FnOnce(&mut [i32]) -> R) -> R {
    thread_local! { static SCRATCH: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) }; }
    SCRATCH.with(|b| {
        let mut b = b.borrow_mut();
        if b.len() < m { b.resize(m, 0); }
        f(&mut b[..m])
    })
}

/// Encoder search effort: how many lag candidates are fitted, which tap counts, how many of the
/// candidates (ranked by the autocorrelation model's predicted saving) are run and priced exactly, and the least saving, in bits
/// per residual value, for which the stage is used at all -- a decode-time trade: each subframe
/// that uses it costs the decoder one more pass, whatever it saves.
#[derive(Debug, Clone, Copy)]
pub struct Search { pub candidates: usize, pub tap_counts: &'static [usize], pub exact: usize, pub min_gain: f64 }

/// The cheapest long-term predictor for residual `e`, as (params, coded residual, bits of coded
/// residual + header), if it saves more than `min_gain` bits per value against `plain_bits` + 1
/// (the residual as it is, plus the "off" flag). `cost` prices a residual exactly
/// (`rice::cost_bits`).
pub fn search(e: &[i64], s: &Search, plain_bits: u64, cost: impl Fn(&[i64]) -> u64) -> Option<(Params, Vec<i64>, u64)> {
    let m = e.len();
    let max_lag = MAX_LAG.min(m / 2);
    if s.candidates == 0 || max_lag < MIN_LAG || m > FFT_MAX / 2 || e.iter().any(|v| v.abs() > LIMIT) { return None; }
    let g14 = crate::prof::span(crate::prof::Phase::LtpFft);
    let r = autocorrelation_f32(e);
    drop(g14);
    let g15 = crate::prof::span(crate::prof::Phase::LtpRankFit);
    // E(T) = sum_{n>=T} e[n-T]^2 = total energy minus the last T samples'.
    let mut tail = vec![0.0f64; m + 1];
    for i in (0..m).rev() { tail[i] = tail[i + 1] + (e[i] as f64) * (e[i] as f64); }
    let mut ranked: Vec<(f64, usize)> = (MIN_LAG..=max_lag).filter_map(|t| {
        let en = tail[0] - tail[m - t];
        (r[t] > 0.0 && en > 0.0).then(|| (r[t] * r[t] / en, t))
    }).collect();
    // Only the best `candidates` are used: select them (a total order, so the same set a full sort
    // gives) and sort just those.
    let by_rank = |a: &(f64, usize), b: &(f64, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    if s.candidates > 0 && ranked.len() > s.candidates {
        ranked.select_nth_unstable_by(s.candidates - 1, by_rank);
        ranked.truncate(s.candidates);
    }
    ranked.sort_by(by_rank);
    // Every candidate is ranked by the saving the autocorrelation model predicts, about
    // m/2 * log2(E / E_left) bits (a Laplacian residual's Rice cost moves with log2 of its
    // spread) less the header, and only the best `exact` are run and priced exactly. Where even
    // the best predicted saving is under half the threshold, nothing is run.
    let e0 = r[0];
    let mut model: Vec<(f64, usize, Vec<i32>)> = Vec::new();
    for &(_, lag) in ranked.iter().take(s.candidates) {
        for &k in s.tap_counts {
            if lag + k / 2 >= m { continue; }
            let Some((taps, left)) = fit(&r, lag, k) else { continue };
            let gain = 0.5 * m as f64 * detmath::log2(e0 / left.max(e0 * 1e-6)) - (1 + LAG_BITS + 2 + TAP_BITS * k as u32) as f64;
            if gain > 0.5 * s.min_gain * m as f64 { model.push((gain, lag, taps)); }
        }
    }
    model.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.len().cmp(&b.2.len())));
    drop(g15);
    let _g11 = crate::prof::span(crate::prof::Phase::LtpPricing);
    let limit = plain_bits as f64 + 1.0 - s.min_gain * m as f64;
    let mut best: Option<(Params, Vec<i64>, u64)> = None;
    for (_, lag, taps) in model.into_iter().take(s.exact.max(1)) {
        let p = Params { lag, taps };
        let coded = forward(&p, e);
        let bits = cost(&coded) + p.header_bits();
        if (bits as f64) < limit && best.as_ref().is_none_or(|b| bits < b.2) { best = Some((p, coded, bits)); }
    }
    best
}

/// Taps for lag `lag` and `k` taps from the Toeplitz approximation of the normal equations
/// (Gram matrix R(|i-j|), right-hand side r(lag + h - i)), quantized, with the energy the model
/// says is left after predicting with them; None if every tap rounds to zero.
fn fit(r: &[f64], lag: usize, k: usize) -> Option<(Vec<i32>, f64)> {
    let h = k / 2;
    let mut a = [[0.0f64; MAX_TAPS + 1]; MAX_TAPS];
    for i in 0..k {
        for j in 0..k { a[i][j] = r[i.abs_diff(j)]; }
        a[i][i] *= 1.0 + 1e-9;
        a[i][k] = r[lag + h - i];
    }
    if !(a[0][0] > 0.0) { return None; }
    for c in 0..k { // Gauss-Jordan with partial pivoting
        let p = (c..k).fold(c, |p, x| if a[x][c].abs() > a[p][c].abs() { x } else { p });
        a.swap(c, p);
        if a[c][c].abs() < 1e-300 { return None; }
        for row in 0..k {
            if row == c { continue; }
            let f = a[row][c] / a[c][c];
            for j in c..=k { a[row][j] -= f * a[c][j]; }
        }
    }
    let taps: Vec<i32> = (0..k).map(|i| {
        let g = a[i][k] / a[i][i] * (1 << SHIFT) as f64;
        (g + if g < 0.0 { -0.5 } else { 0.5 }).trunc().clamp(-(TAP_MAX as f64), TAP_MAX as f64) as i32
    }).collect();
    if taps.iter().all(|&g| g == 0) { return None; }
    // Energy left after prediction with the *quantized* taps, in the same Toeplitz model:
    // E - 2 g.b + g'Ag, with g = taps / 2^SHIFT, b[i] = r[lag + h - i], A[i][j] = r[|i - j|].
    let g: Vec<f64> = taps.iter().map(|&t| t as f64 / (1 << SHIFT) as f64).collect();
    let mut left = r[0];
    for i in 0..k {
        left -= 2.0 * g[i] * r[lag + h - i];
        for j in 0..k { left += g[i] * g[j] * r[i.abs_diff(j)]; }
    }
    Some((taps, left))
}

const FFT_MAX: usize = 1 << 16;

/// Twiddles, from `detmath::cos` so they are the same on every platform. `base[i]` = (cos, -sin)
/// of 2*pi*i/FFT_MAX; the FFT stage of length L reads its `L/2` twiddles contiguously from
/// `cos[L/2..L]` / `sin[L/2..L]` (the same values as `base[q * FFT_MAX / L]`), so its butterflies
/// vectorize.
struct Twiddles { base: Vec<(f64, f64)>, cos: Vec<f64>, sin: Vec<f64> }

fn twiddles() -> &'static Twiddles {
    use std::sync::OnceLock;
    static TW: OnceLock<Twiddles> = OnceLock::new();
    TW.get_or_init(|| {
        let base: Vec<(f64, f64)> = (0..FFT_MAX / 2).map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / FFT_MAX as f64;
            (detmath::cos(a), -detmath::cos(a - std::f64::consts::FRAC_PI_2))
        }).collect();
        let (mut cos, mut sin) = (vec![0.0; FFT_MAX], vec![0.0; FFT_MAX]);
        let mut len = 2;
        while len <= FFT_MAX {
            for q in 0..len / 2 { (cos[len / 2 + q], sin[len / 2 + q]) = base[q * (FFT_MAX / len)]; }
            len <<= 1;
        }
        Twiddles { base, cos, sin }
    })
}

/// Autocorrelation r[t] = sum_n e[n] e[n+t], t < len, by FFT. Everything is `+ - * /` on
/// `detmath` twiddles in a fixed order (no FMA, no reassociation, the same in the AVX2 build), so
/// the result, and every encoder choice made from it, is bit-identical on every platform.
pub fn autocorrelation(e: &[i64]) -> Vec<f64> {
    #[cfg(target_os = "macos")]
    if crate::accel::apple_enabled() { return autocorrelation_vdsp(e); }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: AVX2 confirmed at runtime by `avx2_enabled`.
            return unsafe { autocorrelation_avx2(e) };
        }
    }
    autocorrelation_impl(e)
}

/// The same autocorrelation on Accelerate's complex FFT (`vDSP_fft_zipD`), for the opt-in `--accel
/// apple` (`accel::enable_apple`): not bit-identical to [`autocorrelation`] (its own butterflies and
/// twiddles), so the encoder's near-ties can break differently. `|FFT(e)|^2` then the inverse
/// transform of that real, even power spectrum; sizes are powers of two, `2 * len` up.
#[cfg(target_os = "macos")]
fn autocorrelation_vdsp(e: &[i64]) -> Vec<f64> {
    use std::sync::OnceLock;
    #[repr(C)]
    struct Split { re: *mut f64, im: *mut f64 }
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn vDSP_create_fftsetupD(log2n: usize, radix: i32) -> *mut core::ffi::c_void;
        fn vDSP_fft_zipD(setup: *mut core::ffi::c_void, c: *const Split, stride: isize, log2n: usize, dir: i32);
    }
    // Setups are read-only once created and safe to share between threads; one per size, never freed.
    static SETUPS: [OnceLock<usize>; 32] = [const { OnceLock::new() }; 32];
    let m = e.len();
    let n = (2 * m).next_power_of_two().max(4);
    let log2n = n.trailing_zeros() as usize;
    let setup = *SETUPS[log2n].get_or_init(|| unsafe { vDSP_create_fftsetupD(log2n, 2) } as usize) as *mut core::ffi::c_void;
    let mut re = vec![0f64; n];
    let mut im = vec![0f64; n];
    for (d, &v) in re.iter_mut().zip(e) { *d = v as f64; }
    let z = Split { re: re.as_mut_ptr(), im: im.as_mut_ptr() };
    // Safety: `re`/`im` hold `n = 2^log2n` values and outlive both calls; `setup` was made for `log2n`
    // (a null setup, if allocation failed, is caught by the assert).
    assert!(!setup.is_null(), "vDSP FFT setup failed");
    unsafe {
        vDSP_fft_zipD(setup, &z, 1, log2n, 1);
        for k in 0..n { re[k] = re[k] * re[k] + im[k] * im[k]; im[k] = 0.0; }
        vDSP_fft_zipD(setup, &z, 1, log2n, -1);
    }
    let scale = 1.0 / n as f64;
    re.truncate(m);
    for v in re.iter_mut() { *v *= scale; }
    re
}

/// The baseline-target build of `autocorrelation` (tests check the two agree bit for bit).
pub fn autocorrelation_portable(e: &[i64]) -> Vec<f64> { autocorrelation_impl(e) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn autocorrelation_avx2(e: &[i64]) -> Vec<f64> { autocorrelation_impl(e) }

/// The float type an FFT autocorrelation runs in: `f64`, or `f32` for the encoder's search (twice the
/// lanes per vector and half the memory traffic; its ~1e-6 relative error is far below what ranking
/// lags by `r[t]^2 / energy` and fitting 7-bit taps can resolve). Each type has its own twiddle tables.
trait Fl: Copy
    + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self> + std::ops::Neg<Output = Self>
    + std::ops::AddAssign + std::ops::SubAssign + 'static
{
    const ZERO: Self;
    const HALF: Self;
    const RH: Self;
    fn from_i64(x: i64) -> Self;
    fn to_f64(self) -> f64;
    fn from_usize(x: usize) -> Self;
    /// `cos`/`sin` stage tables (the length-`L` stage's twiddles at `[L/2..L]`).
    fn stage_tables() -> &'static (Vec<Self>, Vec<Self>);
    /// The FFT's long stages; `f32` has an AVX2 version (same operations, so the same bits).
    fn dif_main(re: &mut [Self], im: &mut [Self]);
    fn dit_main(re: &mut [Self], im: &mut [Self]);
    /// `W^k = e^(-2 pi i k / n)` for `k < n / 2`, contiguous; one table per size.
    fn packing(n: usize) -> &'static [(Self, Self)];
    /// [`Fl::packing`] in bit-reversed order: `[pos] = W^rev(pos)`.
    fn packing_rev(n: usize) -> &'static [(Self, Self)];
}

macro_rules! impl_fl {
    ($t:ty, $from_i64:expr, $dif:expr, $dit:expr) => {
        impl Fl for $t {
            const ZERO: Self = 0.0;
            const HALF: Self = 0.5;
            const RH: Self = std::f64::consts::FRAC_1_SQRT_2 as $t;
            #[inline(always)]
            fn from_i64(x: i64) -> Self { ($from_i64)(x) }
            #[inline(always)]
            fn to_f64(self) -> f64 { self as f64 }
            #[inline(always)]
            fn from_usize(x: usize) -> Self { x as $t }
            #[inline(always)]
            fn dif_main(re: &mut [Self], im: &mut [Self]) { ($dif)(re, im) }
            #[inline(always)]
            fn dit_main(re: &mut [Self], im: &mut [Self]) { ($dit)(re, im) }
            fn stage_tables() -> &'static (Vec<Self>, Vec<Self>) {
                use std::sync::OnceLock;
                static T: OnceLock<(Vec<$t>, Vec<$t>)> = OnceLock::new();
                T.get_or_init(|| {
                    let tw = twiddles();
                    (tw.cos.iter().map(|&v| v as $t).collect(), tw.sin.iter().map(|&v| v as $t).collect())
                })
            }
            fn packing(n: usize) -> &'static [(Self, Self)] {
                use std::sync::OnceLock;
                static TABLES: [OnceLock<Vec<($t, $t)>>; 32] = [const { OnceLock::new() }; 32];
                TABLES[n.trailing_zeros() as usize].get_or_init(|| {
                    let (tw, st) = (twiddles(), FFT_MAX / n);
                    (0..n / 2).map(|k| { let (c, s) = tw.base[k * st]; (c as $t, s as $t) }).collect()
                })
            }
            fn packing_rev(n: usize) -> &'static [(Self, Self)] {
                use std::sync::OnceLock;
                static TABLES: [OnceLock<Vec<($t, $t)>>; 32] = [const { OnceLock::new() }; 32];
                TABLES[n.trailing_zeros() as usize].get_or_init(|| {
                    let (wk, rev) = (<$t as Fl>::packing(n), bit_reversal(n / 2));
                    (0..n / 2).map(|pos| wk[rev[pos] as usize]).collect()
                })
            }
        }
    };
}
impl_fl!(f64, |x: i64| x as f64, dif_main_generic::<f64>, dit_main_generic::<f64>);
impl_fl!(f32, |x: i64| x as i32 as f32, dif_main_f32, dit_main_f32);

fn dif_main_f32(re: &mut [f32], im: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: AVX2 confirmed at runtime; `re`/`im` have the same power-of-two length >= 16.
            return unsafe { avx::dif_main(re, im) };
        }
    }
    dif_main_generic::<f32>(re, im)
}

fn dit_main_f32(re: &mut [f32], im: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: as above.
            return unsafe { avx::dit_main(re, im) };
        }
    }
    dit_main_generic::<f32>(re, im)
}

/// AVX2 versions of the long stages, written with intrinsics: the compiler's own vectorization of
/// the generic loops ran at about 3 cycles per butterfly. Multiplies and adds stay separate (no FMA),
/// in the generic code's order, so the results are bit-identical to it.
#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn dif_main(re: &mut [f32], im: &mut [f32]) {
        let n = re.len();
        let (cos, sin) = <f32 as super::Fl>::stage_tables();
        let (rp, ip) = (re.as_mut_ptr(), im.as_mut_ptr());
        let mut len = n;
        while len >= 16 {
            let half = len / 2;
            let (c, s) = (cos.as_ptr().add(half), sin.as_ptr().add(half));
            let mut base = 0;
            while base < n {
                let (r0, i0) = (rp.add(base), ip.add(base));
                let (r1, i1) = (r0.add(half), i0.add(half));
                let mut q = 0;
                while q < half {
                    let (ar, ai) = (_mm256_loadu_ps(r0.add(q)), _mm256_loadu_ps(i0.add(q)));
                    let (br, bi) = (_mm256_loadu_ps(r1.add(q)), _mm256_loadu_ps(i1.add(q)));
                    let (wc, ws) = (_mm256_loadu_ps(c.add(q)), _mm256_loadu_ps(s.add(q)));
                    let (dr, di) = (_mm256_sub_ps(ar, br), _mm256_sub_ps(ai, bi));
                    _mm256_storeu_ps(r0.add(q), _mm256_add_ps(ar, br));
                    _mm256_storeu_ps(i0.add(q), _mm256_add_ps(ai, bi));
                    _mm256_storeu_ps(r1.add(q), _mm256_sub_ps(_mm256_mul_ps(dr, wc), _mm256_mul_ps(di, ws)));
                    _mm256_storeu_ps(i1.add(q), _mm256_add_ps(_mm256_mul_ps(dr, ws), _mm256_mul_ps(di, wc)));
                    q += 8;
                }
                base += len;
            }
            len >>= 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dit_main(re: &mut [f32], im: &mut [f32]) {
        let n = re.len();
        let (cos, sin) = <f32 as super::Fl>::stage_tables();
        let (rp, ip) = (re.as_mut_ptr(), im.as_mut_ptr());
        let zero = _mm256_setzero_ps();
        let mut len = 16;
        while len <= n {
            let half = len / 2;
            let (c, s) = (cos.as_ptr().add(half), sin.as_ptr().add(half));
            let mut base = 0;
            while base < n {
                let (r0, i0) = (rp.add(base), ip.add(base));
                let (r1, i1) = (r0.add(half), i0.add(half));
                let mut q = 0;
                while q < half {
                    let (ar, ai) = (_mm256_loadu_ps(r0.add(q)), _mm256_loadu_ps(i0.add(q)));
                    let (br, bi) = (_mm256_loadu_ps(r1.add(q)), _mm256_loadu_ps(i1.add(q)));
                    let wc = _mm256_loadu_ps(c.add(q));
                    let ws = _mm256_sub_ps(zero, _mm256_loadu_ps(s.add(q))); // -sin
                    let xr = _mm256_sub_ps(_mm256_mul_ps(br, wc), _mm256_mul_ps(bi, ws));
                    let xi = _mm256_add_ps(_mm256_mul_ps(br, ws), _mm256_mul_ps(bi, wc));
                    _mm256_storeu_ps(r1.add(q), _mm256_sub_ps(ar, xr));
                    _mm256_storeu_ps(i1.add(q), _mm256_sub_ps(ai, xi));
                    _mm256_storeu_ps(r0.add(q), _mm256_add_ps(ar, xr));
                    _mm256_storeu_ps(i0.add(q), _mm256_add_ps(ai, xi));
                    q += 8;
                }
                base += len;
            }
            len <<= 1;
        }
    }
}

/// `rev[i]` = `i` with its `log2(n)` low bits reversed, for a power-of-two `n >= 2`; one table per size.
fn bit_reversal(n: usize) -> &'static [u32] {
    use std::sync::OnceLock;
    static TABLES: [OnceLock<Vec<u32>>; 32] = [const { OnceLock::new() }; 32];
    let bits = n.trailing_zeros();
    TABLES[bits as usize].get_or_init(|| (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect())
}

#[inline(always)]
fn autocorrelation_impl(e: &[i64]) -> Vec<f64> { autocorrelation_core::<f64>(e) }

/// [`autocorrelation`] in `f32` (see [`Fl`]): what the encoder's search uses (on macOS with
/// `--accel apple`, Accelerate's FFT as before).
pub fn autocorrelation_f32(e: &[i64]) -> Vec<f64> {
    #[cfg(target_os = "macos")]
    if crate::accel::apple_enabled() { return autocorrelation_vdsp(e); }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: AVX2 confirmed at runtime by `avx2_enabled`.
            return unsafe { autocorrelation_f32_avx2(e) };
        }
    }
    autocorrelation_core::<f32>(e)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn autocorrelation_f32_avx2(e: &[i64]) -> Vec<f64> { autocorrelation_core::<f32>(e) }

/// The baseline-target build of [`autocorrelation_f32`] (tests check the two agree bit for bit).
pub fn autocorrelation_f32_portable(e: &[i64]) -> Vec<f64> { autocorrelation_core::<f32>(e) }

#[inline(always)]
fn autocorrelation_core<F: Fl>(e: &[i64]) -> Vec<f64> {
    // Real input, so one complex FFT of half the length does it: z[j] = e[2j] + i e[2j+1], and
    // the spectrum X of the zero-padded e (length n) is recovered from Z. The power spectrum is
    // real and even, so the inverse uses the same packing the other way round.
    //
    // The spectrum only passes through a pointwise operation, so the forward transform is
    // decimation-in-frequency (natural order in, bit-reversed out) and the inverse decimation-in-time
    // (bit-reversed in, natural out): no bit-reversal permutation at all.
    let m = e.len();
    let n = (2 * m).next_power_of_two().max(4);
    let h = n / 2;
    if h < 16 { return autocorrelation_small(e); }
    let wk = F::packing(n);
    let wr = F::packing_rev(n);
    let mut re = vec![F::ZERO; h];
    let mut im = vec![F::ZERO; h];
    for (j, c) in e.chunks(2).enumerate() {
        re[j] = F::from_i64(c[0]);
        if let Some(&v) = c.get(1) { im[j] = F::from_i64(v); }
    }
    fft_dif(&mut re, &mut im);
    // Spectrum bins sit in bit-reversed order: position `pos` holds bin `k = rev(pos)`, and its
    // partner h - k is at `pos` mirrored inside its dyadic block [2^t, 2^(t+1)) (`pos = 2^t + j`
    // pairs with `2^(t+1) - 1 - j`; positions 0 and 1 are their own partners), so both loops below
    // run sequentially forwards and backwards over blocks, with no table lookups.
    let bin = |zr: F, zi: F, cr: F, ci: F, wc: F, ws: F| -> F {
        let (er, ei) = ((zr + cr) * F::HALF, (zi + ci) * F::HALF); // even samples' spectrum
        let (or, oi) = ((zi - ci) * F::HALF, -(zr - cr) * F::HALF); // odd samples': (Z - conj) / 2i
        let (xr, xi) = (er + wc * or - ws * oi, ei + wc * oi + ws * or);
        xr * xr + xi * xi
    };
    // Power spectrum of bin k = rev(pos), then that of bin h (one more value, from bin 0).
    let mut p = vec![F::ZERO; h];
    p[0] = bin(re[0], im[0], re[0], -im[0], wk[0].0, wk[0].1);
    let p_nyq = bin(re[0], im[0], re[0], -im[0], -F::from_usize(1), F::ZERO); // W^h = -1
    let mut lo = 1;
    while lo < h {
        let (zr, zi) = (&re[lo..2 * lo], &im[lo..2 * lo]);
        let w = &wr[lo..2 * lo];
        let out = &mut p[lo..2 * lo];
        for j in 0..lo {
            let (cr, ci) = (zr[lo - 1 - j], -zi[lo - 1 - j]); // conj(Z[h - k])
            out[j] = bin(zr[j], zi[j], cr, ci, w[j].0, w[j].1);
        }
        lo *= 2;
    }
    // Inverse input: Z' = E + i O per bin, written where the DIT expects it (the same positions).
    let (mut zr_new, mut zi_new) = (vec![F::ZERO; h], vec![F::ZERO; h]);
    {
        // Bin 0 pairs with bin h.
        let (ev, d) = ((p[0] + p_nyq) * F::HALF, (p[0] - p_nyq) * F::HALF);
        let (wc, ws) = wk[0];
        let (or, oi) = (d * wc, -d * ws);
        zr_new[0] = ev - oi;
        zi_new[0] = or;
    }
    let mut lo = 1;
    while lo < h {
        let pp = &p[lo..2 * lo];
        let w = &wr[lo..2 * lo];
        let (zr, zi) = (&mut zr_new[lo..2 * lo], &mut zi_new[lo..2 * lo]);
        for j in 0..lo {
            let (pk, ph) = (pp[j], pp[lo - 1 - j]);
            let (ev, d) = ((pk + ph) * F::HALF, (pk - ph) * F::HALF);
            let (wc, ws) = w[j];
            let (or, oi) = (d * wc, -d * ws); // d / W^k
            zr[j] = ev - oi; // Z = E + i O
            zi[j] = or;
        }
        lo *= 2;
    }
    fft_dit_inverse(&mut zr_new, &mut zi_new);
    let scale = F::from_usize(h);
    let mut out = vec![0.0; m];
    for (j, o) in out.chunks_mut(2).enumerate() {
        o[0] = (zr_new[j] / scale).to_f64();
        if o.len() > 1 { o[1] = (zi_new[j] / scale).to_f64(); }
    }
    out
}

/// The original radix-2 `f64` form, for the few sizes too small for [`autocorrelation_core`]'s unrolled ends.
fn autocorrelation_small(e: &[i64]) -> Vec<f64> {
    let tw = twiddles();
    let m = e.len();
    let n = (2 * m).next_power_of_two().max(4);
    let h = n / 2;
    let st = FFT_MAX / n;
    let mut re = vec![0.0; h];
    let mut im = vec![0.0; h];
    for (j, c) in e.chunks(2).enumerate() {
        re[j] = c[0] as f64;
        if let Some(&v) = c.get(1) { im[j] = v as f64; }
    }
    fft::<false>(&mut re, &mut im, tw);
    let mut p = vec![0.0; h + 1];
    for (k, pk) in p.iter_mut().enumerate() {
        let (zr, zi) = (re[k % h], im[k % h]);
        let (cr, ci) = (re[(h - k) % h], -im[(h - k) % h]);
        let (er, ei) = ((zr + cr) * 0.5, (zi + ci) * 0.5);
        let (or, oi) = ((zi - ci) * 0.5, -(zr - cr) * 0.5);
        let (wc, ws) = if k < h { tw.base[k * st] } else { (-1.0, 0.0) };
        let (xr, xi) = (er + wc * or - ws * oi, ei + wc * oi + ws * or);
        *pk = xr * xr + xi * xi;
    }
    for k in 0..h {
        let (ev, d) = ((p[k] + p[h - k]) * 0.5, (p[k] - p[h - k]) * 0.5);
        let (wc, ws) = tw.base[k * st];
        let (or, oi) = (d * wc, -d * ws);
        re[k] = ev - oi;
        im[k] = or;
    }
    fft::<true>(&mut re, &mut im, tw);
    let mut out = vec![0.0; m];
    for (j, o) in out.chunks_mut(2).enumerate() {
        o[0] = re[j] / h as f64;
        if o.len() > 1 { o[1] = im[j] / h as f64; }
    }
    out
}

/// Decimation-in-frequency forward FFT (`n >= 16`, natural order in, bit-reversed order out): stages
/// of length `n` down to 16 vectorize across the butterflies of a block; the last three (blocks of
/// 8, too short to vectorize) are unrolled.
#[inline(always)]
fn fft_dif<F: Fl>(re: &mut [F], im: &mut [F]) {
    F::dif_main(re, im);
    fft_dif_tail(re, im);
}

/// [`fft_dif`]'s stages of length `n` down to 16.
#[inline(always)]
fn dif_main_generic<F: Fl>(re: &mut [F], im: &mut [F]) {
    let n = re.len();
    let (cos, sin) = F::stage_tables();
    let mut len = n;
    while len >= 16 {
        let half = len / 2;
        let (c, sn) = (&cos[half..len], &sin[half..len]);
        for (rb, ib) in re.chunks_exact_mut(len).zip(im.chunks_exact_mut(len)) {
            let (r0, r1) = rb.split_at_mut(half);
            let (i0, i1) = ib.split_at_mut(half);
            for ((((ar, ai), (br, bi)), &wc), &ws) in r0.iter_mut().zip(i0.iter_mut()).zip(r1.iter_mut().zip(i1.iter_mut())).zip(c).zip(sn) {
                let (xr, xi, yr, yi) = (*ar, *ai, *br, *bi);
                let (dr, di) = (xr - yr, xi - yi);
                *ar = xr + yr; *ai = xi + yi;
                *br = dr * wc - di * ws;
                *bi = dr * ws + di * wc;
            }
        }
        len >>= 1;
    }
}

/// [`fft_dif`]'s last three stages, on blocks of 8.
#[inline(always)]
fn fft_dif_tail<F: Fl>(re: &mut [F], im: &mut [F]) {
    let rh = F::RH;
    for (rb, ib) in re.chunks_exact_mut(8).zip(im.chunks_exact_mut(8)) {
        let (mut xr, mut xi) = ([F::ZERO; 8], [F::ZERO; 8]);
        xr.copy_from_slice(rb);
        xi.copy_from_slice(ib);
        // len 8: (x[q], x[q + 4]) -> (a + b, (a - b) * e^(-2 pi i q / 8)).
        let d: [(F, F); 4] = std::array::from_fn(|q| (xr[q] - xr[q + 4], xi[q] - xi[q + 4]));
        for q in 0..4 { xr[q] += xr[q + 4]; xi[q] += xi[q + 4]; }
        (xr[4], xi[4]) = d[0];
        (xr[5], xi[5]) = (rh * (d[1].0 + d[1].1), rh * (d[1].1 - d[1].0));
        (xr[6], xi[6]) = (d[2].1, -d[2].0);
        (xr[7], xi[7]) = (rh * (d[3].1 - d[3].0), -rh * (d[3].0 + d[3].1));
        // len 4, on each half: (x[q], x[q + 2]) -> (a + b, (a - b) * e^(-2 pi i q / 4)).
        for base in [0usize, 4] {
            let (d0r, d0i) = (xr[base] - xr[base + 2], xi[base] - xi[base + 2]);
            let (d1r, d1i) = (xr[base + 1] - xr[base + 3], xi[base + 1] - xi[base + 3]);
            xr[base] += xr[base + 2]; xi[base] += xi[base + 2];
            xr[base + 1] += xr[base + 3]; xi[base + 1] += xi[base + 3];
            (xr[base + 2], xi[base + 2]) = (d0r, d0i);
            (xr[base + 3], xi[base + 3]) = (d1i, -d1r);
        }
        // len 2.
        for p in 0..4 {
            let (a, b) = (2 * p, 2 * p + 1);
            let (dr, di) = (xr[a] - xr[b], xi[a] - xi[b]);
            xr[a] += xr[b]; xi[a] += xi[b];
            xr[b] = dr; xi[b] = di;
        }
        rb.copy_from_slice(&xr);
        ib.copy_from_slice(&xi);
    }
}

/// Decimation-in-time inverse FFT (unnormalized; `n >= 16`, bit-reversed order in, natural order out):
/// the mirror of [`fft_dif`].
#[inline(always)]
fn fft_dit_inverse<F: Fl>(re: &mut [F], im: &mut [F]) {
    fft_dit_head(re, im);
    F::dit_main(re, im);
}

/// [`fft_dit_inverse`]'s first three stages, on blocks of 8.
#[inline(always)]
fn fft_dit_head<F: Fl>(re: &mut [F], im: &mut [F]) {
    let rh = F::RH;
    for (rb, ib) in re.chunks_exact_mut(8).zip(im.chunks_exact_mut(8)) {
        let (mut xr, mut xi) = ([F::ZERO; 8], [F::ZERO; 8]);
        xr.copy_from_slice(rb);
        xi.copy_from_slice(ib);
        // len 2.
        for p in 0..4 {
            let (a, b) = (2 * p, 2 * p + 1);
            let (sr, si) = (xr[a] + xr[b], xi[a] + xi[b]);
            xr[b] = xr[a] - xr[b]; xi[b] = xi[a] - xi[b];
            xr[a] = sr; xi[a] = si;
        }
        // len 4, on each half: t = x[q + 2] * e^(+2 pi i q / 4); (a + t, a - t).
        for base in [0usize, 4] {
            let (t0r, t0i) = (xr[base + 2], xi[base + 2]);
            let (t1r, t1i) = (-xi[base + 3], xr[base + 3]);
            let (a0r, a0i, a1r, a1i) = (xr[base], xi[base], xr[base + 1], xi[base + 1]);
            (xr[base], xi[base]) = (a0r + t0r, a0i + t0i);
            (xr[base + 2], xi[base + 2]) = (a0r - t0r, a0i - t0i);
            (xr[base + 1], xi[base + 1]) = (a1r + t1r, a1i + t1i);
            (xr[base + 3], xi[base + 3]) = (a1r - t1r, a1i - t1i);
        }
        // len 8: t = x[q + 4] * e^(+2 pi i q / 8).
        let t: [(F, F); 4] = [
            (xr[4], xi[4]),
            (rh * (xr[5] - xi[5]), rh * (xr[5] + xi[5])),
            (-xi[6], xr[6]),
            (-rh * (xr[7] + xi[7]), rh * (xr[7] - xi[7])),
        ];
        for q in 0..4 {
            let (ar, ai) = (xr[q], xi[q]);
            (xr[q], xi[q]) = (ar + t[q].0, ai + t[q].1);
            (xr[q + 4], xi[q + 4]) = (ar - t[q].0, ai - t[q].1);
        }
        rb.copy_from_slice(&xr);
        ib.copy_from_slice(&xi);
    }
}

/// [`fft_dit_inverse`]'s stages of length 16 up to `n`.
#[inline(always)]
fn dit_main_generic<F: Fl>(re: &mut [F], im: &mut [F]) {
    let n = re.len();
    let (cos, sin) = F::stage_tables();
    let mut len = 16;
    while len <= n {
        let half = len / 2;
        let (c, sn) = (&cos[half..len], &sin[half..len]);
        for (rb, ib) in re.chunks_exact_mut(len).zip(im.chunks_exact_mut(len)) {
            let (r0, r1) = rb.split_at_mut(half);
            let (i0, i1) = ib.split_at_mut(half);
            for ((((ar, ai), (br, bi)), &c), &s) in r0.iter_mut().zip(i0.iter_mut()).zip(r1.iter_mut().zip(i1.iter_mut())).zip(c).zip(sn) {
                let (wc, ws) = (c, -s);
                let (xr, xi) = (*br * wc - *bi * ws, *br * ws + *bi * wc);
                *br = *ar - xr; *bi = *ai - xi;
                *ar += xr; *ai += xi;
            }
        }
        len <<= 1;
    }
}

/// Radix-2 FFT (sign of the sine flipped for the inverse, which is left unnormalized). Each
/// stage's butterflies run over contiguous halves of every block with contiguous twiddles.
#[inline(always)]
fn fft<const INVERSE: bool>(re: &mut [f64], im: &mut [f64], tw: &Twiddles) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 { j ^= bit; bit >>= 1; }
        j |= bit;
        if i < j { re.swap(i, j); im.swap(i, j); }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let (c, sn) = (&tw.cos[half..len], &tw.sin[half..len]);
        for (rb, ib) in re.chunks_exact_mut(len).zip(im.chunks_exact_mut(len)) {
            let (r0, r1) = rb.split_at_mut(half);
            let (i0, i1) = ib.split_at_mut(half);
            for q in 0..half {
                let (wc, ws) = (c[q], if INVERSE { -sn[q] } else { sn[q] });
                let (xr, xi) = (r1[q] * wc - i1[q] * ws, r1[q] * ws + i1[q] * wc);
                r1[q] = r0[q] - xr; i1[q] = i0[q] - xi;
                r0[q] += xr; i0[q] += xi;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A corrupted stream can decode a value past `LIMIT`; later predictions then read it and the
    /// `i32` sums overflow. That must end in the `Err` at the end of the subframe -- in debug and
    /// sanitizer builds too, not a "multiply with overflow" panic (fuzz target `chunk_payload`).
    #[test]
    fn corrupted_residuals_overflowing_i32_return_err_not_panic() {
        for k in TAP_COUNTS {
            let p = Params { lag: 40, taps: vec![i32::MAX >> 1; k] };
            let mut e: Vec<i64> = (0..400).map(|i| if i % 3 == 0 { i32::MAX as i64 } else { -(1i64 << 30) }).collect();
            assert!(inverse(&p, &mut e).is_err(), "k={k}");
            let mut e2 = e.clone();
            assert!(inverse_portable(&p, &mut e2).is_err(), "portable k={k}");
        }
    }

    fn lcg(seed: &mut u64) -> u64 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *seed >> 33 }

    #[test]
    fn inverse_undoes_forward() {
        let mut seed = 7u64;
        for trial in 0..400 {
            let m = 40 + (lcg(&mut seed) % 5000) as usize;
            let amp = [3i64, 300, 30000, LIMIT][trial % 4];
            let period = 32 + (lcg(&mut seed) % 400) as usize;
            let e: Vec<i64> = (0..m).map(|i| {
                let v = ((i % period) as i64 * 7919 % (2 * amp + 1)) - amp + (lcg(&mut seed) % 5) as i64 - 2;
                v.clamp(-LIMIT, LIMIT)
            }).collect();
            let k = TAP_COUNTS[trial % 4];
            let lag = MIN_LAG + (lcg(&mut seed) as usize % (MAX_LAG - MIN_LAG + 1));
            let taps: Vec<i32> = (0..k).map(|_| (lcg(&mut seed) % 127) as i32 - 63).collect();
            let p = Params { lag, taps };
            let coded = forward(&p, &e);
            assert_eq!(coded, forward_reference(&p, &e), "trial {trial} (forward)");
            let mut back = coded.clone();
            inverse(&p, &mut back).unwrap();
            assert_eq!(back, e, "trial {trial}");
            let mut back = coded.clone();
            inverse_portable(&p, &mut back).unwrap();
            assert_eq!(back, e, "trial {trial} (portable)");
        }
    }

    #[test]
    fn header_round_trips_and_rejects_short_lags() {
        let p = Params { lag: 1234, taps: vec![-64, 0, 17, 63, -1] };
        let mut w = BitWriter::new();
        Params::write(Some(&p), &mut w);
        Params::write(None, &mut w);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(Params::read(&mut r).unwrap(), Some(p.clone()));
        assert_eq!(Params::read(&mut r).unwrap(), None);
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(31, LAG_BITS);
        w.write_bits(0, 2);
        w.write_bits(1, TAP_BITS);
        let bytes = w.finish();
        assert!(Params::read(&mut BitReader::new(&bytes)).is_err());
    }

    #[test]
    fn inverse_rejects_out_of_range() {
        let p = Params { lag: 32, taps: vec![32] };
        let mut e = vec![LIMIT; 100];
        assert!(inverse(&p, &mut e).is_err()); // 2 * LIMIT after the first lag
        let mut e = vec![0i64; 100];
        e[0] = LIMIT + 1;
        assert!(inverse(&p, &mut e).is_err());
    }

    #[test]
    fn search_finds_a_periodic_residual() {
        let mut seed = 3u64;
        let base: Vec<i64> = (0..157).map(|_| (lcg(&mut seed) % 2001) as i64 - 1000).collect();
        let e: Vec<i64> = (0..8192).map(|i| base[i % 157] + (lcg(&mut seed) % 21) as i64 - 10).collect();
        let cost = |v: &[i64]| crate::rice::cost_bits(v);
        let s = Search { candidates: 4, tap_counts: &TAP_COUNTS, exact: 2, min_gain: 0.0 };
        let (p, coded, bits) = search(&e, &s, cost(&e), cost).expect("periodic residual must use LTP");
        assert_eq!(p.lag % 157, 0);
        assert!(bits < cost(&e) * 3 / 5, "{bits} vs {} ({p:?})", cost(&e));
        let mut back = coded;
        inverse(&p, &mut back).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn f32_autocorrelation_tracks_the_f64_one_and_agrees_across_builds() {
        let mut seed = 5u64;
        for len in [64usize, 999, 1000, 4096, 4097, 20000] {
            // Residual-like values (several hundred, a few outliers) inside the search's `LIMIT`.
            let e: Vec<i64> = (0..len).map(|i| {
                let v = (lcg(&mut seed) % 2001) as i64 - 1000;
                if i % 517 == 0 { v * 500 } else { v }
            }).collect();
            let (r32, r64, rp) = (autocorrelation_f32(&e), autocorrelation(&e), autocorrelation_f32_portable(&e));
            assert_eq!(r32.len(), len);
            assert!(r32.iter().zip(&rp).all(|(a, b)| a.to_bits() == b.to_bits()), "len {len}: dispatch and portable differ");
            let worst = r32.iter().zip(&r64).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / r64[0];
            assert!(worst < 5e-6, "len {len}: f32 differs from f64 by {worst:e} of r[0]");
        }
    }

    #[test]
    fn autocorrelation_matches_direct() {
        let mut seed = 11u64;
        for len in [1usize, 2, 3, 999, 1000, 4097] {
            let e: Vec<i64> = (0..len).map(|_| (lcg(&mut seed) % 2001) as i64 - 1000).collect();
            let r = autocorrelation(&e);
            assert_eq!(r.len(), len);
            let rp = autocorrelation_portable(&e);
            assert!(r.iter().zip(&rp).all(|(a, b)| a.to_bits() == b.to_bits()), "len {len}: dispatch and portable differ");
            #[cfg(target_os = "macos")]
            {
                let rv = autocorrelation_vdsp(&e);
                assert_eq!(rv.len(), len);
                assert!(rv.iter().zip(&r).all(|(a, b)| (a - b).abs() <= 1e-9 * r[0].max(1.0)), "len {len}: vdsp differs from the FFT");
            }
            for t in [0usize, 1, 2, 32, 500, 998, 999, 4096].into_iter().filter(|&t| t < len) {
                let d: i64 = (0..e.len() - t).map(|n| e[n] * e[n + t]).sum();
                assert!((r[t] - d as f64).abs() <= 1e-6 * r[0], "len {len} lag {t}: {} vs {d}", r[t]);
            }
        }
    }
}
