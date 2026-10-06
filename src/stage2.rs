//! Second-stage adaptive prediction on block-mode residuals (format v14, research).
//!
//! After LPC (and cross-channel) prediction, a subframe's residual may still carry structure the
//! frame-wide LPC coefficients cannot follow: slow spectral change inside the frame, long
//! correlations beyond LPC order 32. This stage predicts each residual from the previous `taps`
//! residuals with a sign-error LMS filter that adapts sample by sample, and codes what is left.
//! The encoder enables it per subframe only where it makes the subframe smaller (1 flag bit
//! otherwise), so it can never cost more than that bit.
//!
//! Arithmetic (exact, identical on every platform):
//! * input scaling: `x = sat16(r >> s)` for `s >= 0`, `sat16(r << -s)` for `s < 0`, with `r`
//!   first clamped to +-2^40 (so the left shift cannot overflow);
//! * prediction: `dot = sum(w[i] * x[i])` in wrapping 32-bit arithmetic over 16-bit weights and
//!   inputs (what `pmaddwd`/`smlal` compute), `pred = (dot << s) >> 14` for `s >= 0`, else
//!   `dot >> (14 - s)` (arithmetic shifts, in 64 bits);
//! * coding: `e = r - pred` (encoder), `r = e + pred` (decoder);
//! * update: `w[i] = sat16(w[i] + sign(e) * (x[i] >> k))`, then the window slides by one with the
//!   new input `x(r)`. Weights and window start at zero in every subframe.
//! The window is ordered oldest first; `w[taps-1]` pairs with the newest input.
use crate::bitio::{BitReader, BitReaderError, BitWriter};

/// Filter lengths selectable per subframe (2-bit code). Multiples of 16 (one AVX2 register).
pub const TAPS: [usize; 4] = [16, 32, 128, 256];
const W_SHIFT: i32 = 14;
const S_MIN: i32 = -16;
const S_MAX: i32 = 31;
/// Residuals beyond this magnitude cannot come from a valid stream (25-bit samples, bounded
/// predictors); a decoder rejects them rather than feeding them on.
pub const MAX_RESIDUAL: i64 = 1 << 40;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params { pub taps: usize, pub k: u32, pub s: i32 }

impl Params {
    /// Header: 1 bit enabled; then 2 bits taps code, 4 bits k (1..=15), 6 bits s + 16 (0..=47).
    pub fn write(p: Option<&Params>, w: &mut BitWriter) {
        let Some(p) = p else { w.write_bits(0, 1); return };
        w.write_bits(1, 1);
        w.write_bits(TAPS.iter().position(|&t| t == p.taps).expect("valid taps") as u64, 2);
        w.write_bits(p.k as u64, 4);
        w.write_bits((p.s - S_MIN) as u64, 6);
    }
    pub fn read(r: &mut BitReader) -> Result<Option<Params>, String> {
        let e = |e: BitReaderError| e.0.to_string();
        if r.read_bits(1).map_err(e)? == 0 { return Ok(None); }
        let taps = TAPS[r.read_bits(2).map_err(e)? as usize];
        let k = r.read_bits(4).map_err(e)? as u32;
        let s = r.read_bits(6).map_err(e)? as i32 + S_MIN;
        if k == 0 || s > S_MAX { return Err(format!("invalid stage-2 parameters k={k} s={s} (corrupted stream?)")); }
        Ok(Some(Params { taps, k, s }))
    }
    pub const HEADER_BITS: u64 = 13;

    /// Input scale for a residual block: its mean magnitude lands near 2^`target`, the range the
    /// 16-bit filter adapts well in .
    pub fn for_block(res: &[i64], taps: usize, k: u32, target: u32) -> Params {
        let mean = res.iter().map(|e| e.unsigned_abs().min(MAX_RESIDUAL as u64)).sum::<u64>() / res.len().max(1) as u64;
        let s = ((64 - mean.leading_zeros()) as i32 - target as i32).clamp(S_MIN, S_MAX);
        Params { taps, k, s }
    }
}

#[inline(always)]
fn input(r: i64, s: i32) -> i16 {
    let r = r.clamp(-MAX_RESIDUAL, MAX_RESIDUAL);
    (if s >= 0 { r >> s } else { r << -s }).clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

#[inline(always)]
fn predict(dot: i32, s: i32) -> i64 {
    if s >= 0 { ((dot as i64) << s) >> W_SHIFT } else { (dot as i64) >> (W_SHIFT - s) }
}

/// Sliding window over the filter inputs: `buf[pos - taps .. pos]` is the current window. Pushing
/// writes at `pos`; when the buffer is full the last `taps` inputs move to the front.
struct Window { buf: Vec<i16>, pos: usize, taps: usize }
impl Window {
    fn new(taps: usize) -> Self { Window { buf: vec![0; taps + 4096], pos: taps, taps } }
    #[inline(always)]
    fn get(&self) -> &[i16] { &self.buf[self.pos - self.taps..self.pos] }
    #[inline(always)]
    fn push(&mut self, x: i16) {
        if self.pos == self.buf.len() {
            self.buf.copy_within(self.pos - self.taps..self.pos, 0);
            self.pos = self.taps;
        }
        self.buf[self.pos] = x;
        self.pos += 1;
    }
}

/// Scalar reference kernels (the definition the SIMD versions must match exactly).
fn dot_scalar(w: &[i16], x: &[i16]) -> i32 {
    w.iter().zip(x).fold(0i32, |a, (&w, &x)| a.wrapping_add(w as i32 * x as i32))
}
fn update_scalar(w: &mut [i16], x: &[i16], sign: i16, k: u32) {
    for (w, &x) in w.iter_mut().zip(x) { *w = w.saturating_add(sign * (x >> k)); }
}

/// Lane-parallel dot product for the carried filter; wrapping i32 sums are order-independent, so this
/// equals `dot_scalar` exactly while letting the compiler emit pmaddwd-style code.
#[inline(always)]
fn carried_dot(w: &[i16], x: &[i16]) -> i32 {
    let mut acc = [0i32; 16];
    let (wc, xc) = (w.chunks_exact(16), x.chunks_exact(16));
    let (wr, xr) = (wc.remainder(), xc.remainder());
    for (a, b) in wc.zip(xc) { for i in 0..16 { acc[i] = acc[i].wrapping_add(a[i] as i32 * b[i] as i32); } }
    let mut t = acc.iter().fold(0i32, |s, &v| s.wrapping_add(v));
    for (&a, &b) in wr.iter().zip(xr) { t = t.wrapping_add(a as i32 * b as i32); }
    t
}
/// AVX2 dot product with `vpmaddwd` pair sums in wrapping i32. Wrapping sums are order-independent (they are sums modulo
/// 2^32), and the one pair sum that overflows `vpmaddwd` (two products of -32768 * -32768) wraps to the same value as
/// the scalar wrapping adds, so this equals `carried_dot` exactly. The compiler's own vectorisation of `carried_dot`
/// used `vpmovsxwd` + `vpmulld` instead (H260).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn carried_dot_avx2(w: &[i16], x: &[i16]) -> i32 {
    use std::arch::x86_64::*;
    let n = w.len().min(x.len());
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    let (mut a0, mut a1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let mut i = 0;
    while i + 32 <= n {
        let w0 = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        let x0 = _mm256_loadu_si256(xp.add(i) as *const __m256i);
        let w1 = _mm256_loadu_si256(wp.add(i + 16) as *const __m256i);
        let x1 = _mm256_loadu_si256(xp.add(i + 16) as *const __m256i);
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, x1));
        i += 32;
    }
    if i + 16 <= n {
        let w0 = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        let x0 = _mm256_loadu_si256(xp.add(i) as *const __m256i);
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
        i += 16;
    }
    let a = _mm256_add_epi32(a0, a1);
    let s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    let mut t = _mm_cvtsi128_si32(s);
    while i < n {
        t = t.wrapping_add(w[i] as i32 * x[i] as i32);
        i += 1;
    }
    t
}
/// The dot product the carried filter uses: the AVX2 kernel inside the AVX2 clone (`AVX`), the portable one elsewhere.
#[inline(always)]
fn carried_dot_sel<const AVX: bool>(w: &[i16], x: &[i16]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if AVX {
        // Safety: `AVX` is only set by `Carried::run_avx2`, which is only called after AVX2 was detected.
        return unsafe { carried_dot_avx2(w, x) };
    }
    carried_dot(w, x)
}
/// Bucketed sign-sign update, `w += sign * ((d << 8) >> k)` clamped to +-32767, in i16 lanes.
#[inline(always)]
fn carried_update(w: &mut [i16], d: &[i16], sign: i16, k: u32) {
    for (w, &d) in w.iter_mut().zip(d) { *w = w.saturating_add(sign * ((d << 8) >> k)).max(-32767); }
}

/// The per-sample loop, shared by both directions: `INVERSE = false` turns residuals into
/// stage-2 residuals in place (encoder), `true` undoes it (decoder).
#[inline(always)]
fn run<const INVERSE: bool>(p: &Params, data: &mut [i64], dot: impl Fn(&[i16], &[i16]) -> i32, update: impl Fn(&mut [i16], &[i16], i16, u32)) -> Result<(), &'static str> {
    let mut w = vec![0i16; p.taps];
    let mut win = Window::new(p.taps);
    for v in data.iter_mut() {
        let pred = predict(dot(&w, win.get()), p.s);
        let (r, e) = if INVERSE {
            let e = *v;
            let r = e.checked_add(pred).filter(|r| r.abs() <= MAX_RESIDUAL).ok_or("stage-2 residual out of range (corrupted stream?)")?;
            *v = r;
            (r, e)
        } else {
            let r = *v;
            let e = r - pred;
            *v = e;
            (r, e)
        };
        let sign = e.signum() as i16;
        if sign != 0 { update(&mut w, win.get(), sign, p.k); }
        win.push(input(r, p.s));
    }
    Ok(())
}

/// Encoder direction. `res` must be within +-`MAX_RESIDUAL` (always true for real predictors).
pub fn forward(p: &Params, res: &mut [i64]) { dispatch::<false>(p, res).expect("forward cannot fail"); }

/// Encoder direction for several candidates on the same block: `out[c]` is what `forward(&ps[c], ..)`
/// leaves. The filter is one serial dependency chain per candidate, so running the candidates side
/// by side (they share the input window, which depends only on the block and `s`) fills the
/// latency the single chain leaves idle. Bit-identical to calling `forward` per candidate.
pub fn forward_multi(ps: &[Params], res: &[i64]) -> Vec<Vec<i64>> {
    #[cfg(target_arch = "aarch64")]
    {
        if ps.len() > 1 && ps.iter().all(|p| p.s == ps[0].s) {
            // Safety: NEON is part of the aarch64 baseline.
            return unsafe { run_multi_neon(ps, res) };
        }
    }
    ps.iter().map(|p| { let mut c = res.to_vec(); forward(p, &mut c); c }).collect()
}

/// `run_neon`'s fused kernel for several candidates sharing one input window (see `forward_multi`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn run_multi_neon(ps: &[Params], res: &[i64]) -> Vec<Vec<i64>> {
    use std::arch::aarch64::*;
    const LOOKBACK: usize = 8;
    let maxt = ps.iter().map(|p| p.taps).max().unwrap_or(0);
    let sh_s = ps[0].s;
    let mut ws: Vec<Vec<i16>> = ps.iter().map(|p| vec![0i16; p.taps]).collect();
    let mut pending = vec![0i16; ps.len()];
    let ks: Vec<int16x8_t> = ps.iter().map(|p| vdupq_n_s16(-(p.k as i16))).collect();
    let mut outs: Vec<Vec<i64>> = ps.iter().map(|_| res.to_vec()).collect();
    let mut buf = vec![0i16; maxt + LOOKBACK + 4096];
    let mut pos = maxt + LOOKBACK;
    let mut tail = vdupq_n_s16(0);
    let mut prev_tail = vdupq_n_s16(0);
    for (i, &r) in res.iter().enumerate() {
        for (c, p) in ps.iter().enumerate() {
            let taps = p.taps;
            let blocks = taps / 8;
            let sg = vdupq_n_s16(pending[c]);
            let upd = pending[c] != 0;
            let mut acc = vdupq_n_s32(0);
            let wp = ws[c].as_mut_ptr();
            for j in 0..blocks {
                let mut wj = vld1q_s16(wp.add(8 * j));
                let last = j + 1 == blocks;
                if upd {
                    let prev = if last { prev_tail } else { vld1q_s16(buf.as_ptr().add(pos - 1 - taps + 8 * j)) };
                    wj = vqaddq_s16(wj, vmulq_s16(vshlq_s16(prev, ks[c]), sg));
                    vst1q_s16(wp.add(8 * j), wj);
                }
                let cur = if last { tail } else { vld1q_s16(buf.as_ptr().add(pos - taps + 8 * j)) };
                acc = vmlal_s16(acc, vget_low_s16(wj), vget_low_s16(cur));
                acc = vmlal_high_s16(acc, wj, cur);
            }
            let e = r - predict(vaddvq_s32(acc), sh_s);
            outs[c][i] = e;
            pending[c] = e.signum() as i16;
        }
        let x = input(r, sh_s);
        prev_tail = tail;
        tail = vextq_s16(tail, vdupq_n_s16(x), 1);
        if pos == buf.len() {
            buf.copy_within(pos - maxt - LOOKBACK..pos, 0);
            pos = maxt + LOOKBACK;
        }
        buf[pos] = x;
        pos += 1;
    }
    outs
}

/// Decoder direction; errors on a corrupted stream that would push residuals out of range.
pub fn inverse(p: &Params, e: &mut [i64]) -> Result<(), &'static str> { dispatch::<true>(p, e) }

/// Scalar-only versions, for tests and timing comparisons.
pub fn forward_scalar(p: &Params, res: &mut [i64]) { run::<false>(p, res, dot_scalar, update_scalar).expect("forward cannot fail"); }
pub fn inverse_scalar(p: &Params, e: &mut [i64]) -> Result<(), &'static str> { run::<true>(p, e, dot_scalar, update_scalar) }

fn dispatch<const INVERSE: bool>(p: &Params, data: &mut [i64]) -> Result<(), &'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: AVX2 confirmed at runtime by `avx2_enabled`.
            return unsafe { run_avx2::<INVERSE>(p, data) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is part of the aarch64 baseline.
        return unsafe { run_neon::<INVERSE>(p, data) };
    }
    #[allow(unreachable_code)]
    run::<INVERSE>(p, data, dot_scalar, update_scalar)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn run_avx2<const INVERSE: bool>(p: &Params, data: &mut [i64]) -> Result<(), &'static str> {
    use std::arch::x86_64::*;
    // Same arithmetic as `run`, reorganized for latency (the filter is a serial chain: each
    // prediction needs the previous sample). Sample n applies the previous sample's pending
    // weight update and computes its own dot product in one pass over the weights; and the newest
    // 16 inputs live in a register (`tail`, with `prev_tail` one sample older), so no vector load
    // reads an input stored a moment ago (a narrow store followed by a wide load of the same bytes
    // cannot be forwarded and stalls). pmaddwd pair sums wrap exactly like the scalar wrapping
    // sum, psignw on (x >> k) with k >= 1 cannot overflow, paddsw saturates like the scalar.
    const LOOKBACK: usize = 16;
    let taps = p.taps;
    let blocks = taps / 16;
    let mut w = vec![0i16; taps];
    // buf[pos - taps .. pos] is the window; one extra block before it holds the previous window.
    let mut buf = vec![0i16; taps + LOOKBACK + 4096];
    let mut pos = taps + LOOKBACK;
    let mut tail = _mm256_setzero_si256();
    let mut prev_tail = _mm256_setzero_si256();
    let mut pending: i16 = 0; // sign(e) of the previous sample: its update is not applied yet
    let kk = _mm_cvtsi32_si128(p.k as i32);
    for v in data.iter_mut() {
        let sg = _mm256_set1_epi16(pending);
        let mut acc = _mm256_setzero_si256();
        let wp = w.as_mut_ptr() as *mut __m256i;
        for j in 0..blocks {
            let mut wj = _mm256_loadu_si256(wp.add(j));
            let last = j + 1 == blocks;
            if pending != 0 {
                let prev = if last { prev_tail } else { _mm256_loadu_si256(buf.as_ptr().add(pos - 1 - taps + 16 * j) as *const __m256i) };
                wj = _mm256_adds_epi16(wj, _mm256_sign_epi16(_mm256_sra_epi16(prev, kk), sg));
                _mm256_storeu_si256(wp.add(j), wj);
            }
            let cur = if last { tail } else { _mm256_loadu_si256(buf.as_ptr().add(pos - taps + 16 * j) as *const __m256i) };
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wj, cur));
        }
        let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        let pred = predict(_mm_cvtsi128_si32(s), p.s);
        let (r, e) = if INVERSE {
            let e = *v;
            let r = e.checked_add(pred).filter(|r| r.abs() <= MAX_RESIDUAL).ok_or("stage-2 residual out of range (corrupted stream?)")?;
            *v = r;
            (r, e)
        } else {
            let r = *v;
            *v = r - pred;
            (r, r - pred)
        };
        pending = e.signum() as i16;
        let x = input(r, p.s);
        // Slide: drop the oldest of the 16, append x as the newest.
        prev_tail = tail;
        let hi = _mm256_permute2x128_si256(tail, tail, 0x81);
        tail = _mm256_insert_epi16(_mm256_alignr_epi8(hi, tail, 2), x, 15);
        if pos == buf.len() {
            buf.copy_within(pos - taps - LOOKBACK..pos, 0);
            pos = taps + LOOKBACK;
        }
        buf[pos] = x;
        pos += 1;
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn run_neon<const INVERSE: bool>(p: &Params, data: &mut [i64]) -> Result<(), &'static str> {
    use std::arch::aarch64::*;
    // The AVX2 kernel's arrangement on NEON (see `run_avx2`): sample n applies the previous
    // sample's pending weight update and computes its own dot product in one pass over the
    // weights, and the newest 8 inputs live in a register (`tail`, `prev_tail` one sample older) so
    // no wide load reads bytes a narrow store just wrote. smlal wraps like the scalar sum,
    // sshl by -k on x then a multiply by +-1 cannot overflow, sqadd saturates like the scalar.
    const LOOKBACK: usize = 8;
    let taps = p.taps;
    let blocks = taps / 8;
    let mut w = vec![0i16; taps];
    let mut buf = vec![0i16; taps + LOOKBACK + 4096];
    let mut pos = taps + LOOKBACK;
    let mut tail = vdupq_n_s16(0);
    let mut prev_tail = vdupq_n_s16(0);
    let mut pending: i16 = 0;
    let sh = vdupq_n_s16(-(p.k as i16));
    for v in data.iter_mut() {
        let sg = vdupq_n_s16(pending);
        let mut acc = vdupq_n_s32(0);
        let wp = w.as_mut_ptr();
        for j in 0..blocks {
            let mut wj = vld1q_s16(wp.add(8 * j));
            let last = j + 1 == blocks;
            if pending != 0 {
                let prev = if last { prev_tail } else { vld1q_s16(buf.as_ptr().add(pos - 1 - taps + 8 * j)) };
                wj = vqaddq_s16(wj, vmulq_s16(vshlq_s16(prev, sh), sg));
                vst1q_s16(wp.add(8 * j), wj);
            }
            let cur = if last { tail } else { vld1q_s16(buf.as_ptr().add(pos - taps + 8 * j)) };
            acc = vmlal_s16(acc, vget_low_s16(wj), vget_low_s16(cur));
            acc = vmlal_high_s16(acc, wj, cur);
        }
        let pred = predict(vaddvq_s32(acc), p.s);
        let (r, e) = if INVERSE {
            let e = *v;
            let r = e.checked_add(pred).filter(|r| r.abs() <= MAX_RESIDUAL).ok_or("stage-2 residual out of range (corrupted stream?)")?;
            *v = r;
            (r, e)
        } else {
            let r = *v;
            *v = r - pred;
            (r, r - pred)
        };
        pending = e.signum() as i16;
        let x = input(r, p.s);
        prev_tail = tail;
        tail = vextq_s16(tail, vdupq_n_s16(x), 1);
        if pos == buf.len() {
            buf.copy_within(pos - taps - LOOKBACK..pos, 0);
            pos = taps + LOOKBACK;
        }
        buf[pos] = x;
        pos += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *seed >> 33 }

    fn signals() -> Vec<Vec<i64>> {
        let mut seed = 7;
        let mut v = Vec::new();
        // AR(2)-ish residual with slow drift, at several amplitudes.
        for amp in [3i64, 200, 30_000, 1 << 22] {
            let mut x = vec![0i64; 5000];
            for n in 2..x.len() {
                let noise = (lcg(&mut seed) % 2001) as i64 - 1000;
                x[n] = (x[n - 1] * 3 / 4 - x[n - 2] / 3 + noise * amp / 1000).clamp(-MAX_RESIDUAL, MAX_RESIDUAL);
            }
            v.push(x);
        }
        // Extremes: saturating inputs and weights, alternating full scale, zeros.
        v.push((0..3000).map(|i| if i % 2 == 0 { MAX_RESIDUAL } else { -MAX_RESIDUAL }).collect());
        v.push((0..3000).map(|i| if (i / 7) % 2 == 0 { 32767 } else { -32768 }).collect());
        v.push(vec![0; 1000]);
        v.push(vec![]);
        v
    }

    #[test]
    fn inverse_undoes_forward_and_simd_matches_scalar() {
        for sig in signals() {
            for &taps in &TAPS {
                for (k, target) in [(1, 9), (6, 9), (15, 12), (4, 3)] {
                    let p = Params::for_block(&sig, taps, k, target);
                    for s in [p.s, S_MIN, S_MAX, 0] {
                        let p = Params { s, ..p };
                        let mut a = sig.clone();
                        forward_scalar(&p, &mut a);
                        let mut b = sig.clone();
                        forward(&p, &mut b);
                        assert_eq!(a, b, "dispatched forward differs from scalar (taps {taps} k {k} s {s})");
                        let mut c = a.clone();
                        inverse(&p, &mut c).unwrap();
                        assert_eq!(c, sig, "round trip (taps {taps} k {k} s {s})");
                        let mut d = a.clone();
                        inverse_scalar(&p, &mut d).unwrap();
                        assert_eq!(d, sig);
                    }
                }
            }
        }
    }

    #[test]
    fn multi_matches_single() {
        for sig in signals() {
            let ps: Vec<Params> = [(256, 6), (256, 5), (128, 5), (32, 4)].iter().map(|&(t, k)| Params::for_block(&sig, t, k, 9)).collect();
            let m = forward_multi(&ps, &sig);
            for (p, got) in ps.iter().zip(&m) { let mut e = sig.clone(); forward(p, &mut e); assert_eq!(&e, got); }
        }
    }

    #[test]
    fn header_round_trips_and_rejects_invalid() {
        for p in [None, Some(Params { taps: 256, k: 6, s: -3 }), Some(Params { taps: 16, k: 15, s: S_MAX }), Some(Params { taps: 128, k: 1, s: S_MIN })] {
            let mut w = BitWriter::new();
            Params::write(p.as_ref(), &mut w);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(Params::read(&mut r).unwrap(), p);
        }
        // k = 0 and s beyond S_MAX are rejected.
        for (k, s_code) in [(0u64, 16u64), (6, 48), (6, 63)] {
            let mut w = BitWriter::new();
            w.write_bits(1, 1); w.write_bits(3, 2); w.write_bits(k, 4); w.write_bits(s_code, 6);
            let bytes = w.finish();
            assert!(Params::read(&mut BitReader::new(&bytes)).is_err());
        }
    }

    #[test]
    fn hostile_stage2_residuals_error_not_panic() {
        let p = Params { taps: 16, k: 1, s: S_MAX };
        let mut e = vec![MAX_RESIDUAL; 100];
        assert!(inverse(&p, &mut e).is_err() || e.iter().all(|r| r.abs() <= MAX_RESIDUAL));
        let mut e = vec![i64::MAX, i64::MIN, 5];
        assert!(inverse(&p, &mut e).is_err());
    }
}

/// Long-filter state carried across the frames of a chunk (H135/H139): one per subframe slot. Unlike
/// [`Params`] filters it is never reset per subframe; the weights, the bucket average and the last
/// `taps` pre-stage-2 residuals survive, and the window is re-quantised with each frame's own `s`.
/// The decoder must advance it on every subframe's recovered residual to stay in sync with the encoder.
/// Update steps are bucketed by input magnitude relative to a running average (sign-sign LMS).
#[derive(Debug, Clone)]
pub struct Carried { taps: usize, k: u32, wide: bool, avg: i64, w: Vec<i16>, hist: Vec<i64> }

impl Carried {
    pub const TAPS: usize = 512;
    pub const K: u32 = 8;
    pub const TARGET: u32 = 9;
    pub fn write_s(s: i32, w: &mut BitWriter) { w.write_bits((s - S_MIN) as u64, 6); }
    pub fn read_s(r: &mut BitReader) -> Result<i32, String> {
        let s = r.read_bits(6).map_err(|e| e.0.to_string())? as i32 + S_MIN;
        if s > S_MAX { return Err(format!("invalid carried stage-2 shift {s} (corrupted stream?)")); }
        Ok(s)
    }
    /// Chunk config byte -> (taps, update shift): 1 = 512n:8, 2 = 1024n:9 (H135).
    pub fn config(cfg: u8) -> Option<(usize, u32)> { match cfg { 1 => Some((512, 8)), 2 => Some((1024, 9)), _ => None } }
    pub fn taps(&self) -> usize { self.taps }
    pub fn k(&self) -> u32 { self.k }
    pub fn new(taps: usize, k: u32) -> Self { Carried { taps, k, wide: false, avg: 0, w: vec![0; taps], hist: vec![0; taps] } }

    /// Config-3 (OLS chunk) variant (H189/H190): wider buckets and a 1/32 average, -0.29% on the whiter 16-bit OLS residual;
    /// 24-bit keeps the narrow bucket (H198: wide cost kirtans +0.14%).
    pub fn new_ols(taps: usize, k: u32, bits_per_sample: u8) -> Self { Carried { wide: bits_per_sample <= 16, ..Self::new(taps, k) } }

    fn bucket(&mut self, x: i16) -> i16 {
        let m = (x as i64).abs();
        if self.wide {
            let b = if x == 0 { 0 } else if m * 96 > self.avg * 16 { 4 } else if m * 96 > self.avg * 6 { 2 } else { 1 };
            self.avg += m - self.avg / 32;
            return x.signum() * b;
        }
        let b = if x == 0 { 0 } else if m * 48 > self.avg * 4 { 4 } else if m * 48 > self.avg { 2 } else { 1 };
        self.avg += m - self.avg / 16;
        x.signum() * b
    }

    fn run<const INVERSE: bool>(&mut self, data: &mut [i64], s: i32) -> Result<(), &'static str> {
        #[cfg(target_arch = "x86_64")]
        if crate::simd::avx2_enabled() {
            // Safety: AVX2 confirmed at runtime. Integer-only code (wrapping dot products, saturating updates), so the
            // clone's results are identical to the portable ones; the tests compare outputs and state.
            return unsafe { self.run_avx2::<INVERSE>(data, s) };
        }
        self.run_impl::<INVERSE, false>(data, s)
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn run_avx2<const INVERSE: bool>(&mut self, data: &mut [i64], s: i32) -> Result<(), &'static str> { self.run_impl::<INVERSE, true>(data, s) }
    #[inline(always)]
    fn run_impl<const INVERSE: bool, const AVX: bool>(&mut self, data: &mut [i64], s: i32) -> Result<(), &'static str> {
        let taps = self.taps;
        let mut win: Vec<i16> = self.hist.iter().map(|&r| input(r, s)).collect();
        let mut adw: Vec<i16> = Vec::with_capacity(taps + data.len());
        for &x in &win { let b = self.bucket(x); adw.push(b); }
        win.reserve(data.len());
        for v in data.iter_mut() {
            let pred = predict(carried_dot_sel::<AVX>(&self.w, &win[win.len() - taps..]), s);
            let (r, e) = if INVERSE {
                let e = *v;
                let r = e.checked_add(pred).filter(|r| r.abs() <= MAX_RESIDUAL).ok_or("stage-2 residual out of range (corrupted stream?)")?;
                *v = r;
                (r, e)
            } else {
                let r = *v;
                let e = r - pred;
                *v = e;
                (r, e)
            };
            let sign = e.signum() as i32;
            if sign != 0 {
                let d = &adw[adw.len() - taps..];
                carried_update(&mut self.w, d, sign as i16, self.k);
            }
            let xi = input(r, s);
            win.push(xi);
            let b = self.bucket(xi);
            adw.push(b);
            self.hist.push(r);
        }
        let cut = self.hist.len() - taps;
        self.hist.drain(..cut);
        Ok(())
    }

    /// Encoder direction: residuals become stage-2 residuals in place.
    pub fn forward(&mut self, res: &mut [i64], s: i32) { self.run::<false>(res, s).expect("forward cannot fail"); }
    /// Decoder direction for a subframe that does not use the carried filter: the pre-stage-2 residual `r`
    /// is already known, so only the state is advanced (same update as `forward`, output discarded).
    pub fn advance(&mut self, r: &[i64], s: i32) { let mut t = r.to_vec(); self.forward(&mut t, s); }
    /// Decoder direction.
    pub fn inverse(&mut self, e: &mut [i64], s: i32) -> Result<(), &'static str> { self.run::<true>(e, s) }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod carried_tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    fn next(seed: &mut u64) -> u64 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *seed >> 33 }

    /// Residual-like test data: noise at several scales, a decaying tone, saturating extremes, zeros.
    #[cfg(target_arch = "x86_64")]
    fn data(kind: usize, len: usize, seed: u64) -> Vec<i64> {
        let mut sd = seed;
        (0..len).map(|t| {
            let r = next(&mut sd) as i64;
            match kind {
                0 => (r % 2001) - 1000,
                1 => (r % 200_001) - 100_000,
                2 => ((t as f64 * 0.05).sin() * 30_000.0) as i64 + (r % 65) - 32,
                3 => if (t / 5) % 2 == 0 { MAX_RESIDUAL } else { -MAX_RESIDUAL },
                _ => if t % 7 == 0 { r % 9 - 4 } else { 0 },
            }
        }).collect()
    }

    /// The `vpmaddwd` dot product equals the scalar wrapping sum for every length (tails included) and for the extremes,
    /// among them the all `-32768` case in which a pair sum overflows `vpmaddwd` itself.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn madd_dot_equals_the_scalar_dot() {
        if !std::is_x86_feature_detected!("avx2") { return; }
        let mut sd = 5u64;
        for n in (0..200).chain([511, 512, 513, 1023, 1024, 1025]) {
            for kind in 0..4 {
                let gen = |sd: &mut u64| -> Vec<i16> { (0..n).map(|_| match kind { 0 => next(sd) as i16, 1 => i16::MIN, 2 => if next(sd) % 2 == 0 { i16::MIN } else { i16::MAX }, _ => (next(sd) % 7) as i16 - 3 }).collect() };
                let (w, x) = (gen(&mut sd), gen(&mut sd));
                let want = w.iter().zip(&x).fold(0i32, |a, (&p, &q)| a.wrapping_add(p as i32 * q as i32));
                assert_eq!(unsafe { carried_dot_avx2(&w, &x) }, want, "n {n} kind {kind}");
                assert_eq!(carried_dot(&w, &x), want, "portable dot, n {n} kind {kind}");
            }
        }
    }

    /// The AVX2 clone of the carried filter must reproduce the portable code exactly: outputs of both directions,
    /// and the filter state (weights, history, bucket average) after every chunk.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn carried_avx2_clone_matches_the_portable_code() {
        if !std::is_x86_feature_detected!("avx2") { return; }
        for (taps, k, bits) in [(512usize, 8u32, 16u8), (512, 8, 24), (1024, 9, 24), (1024, 9, 16), (16, 3, 16), (48, 5, 24)] {
            for kind in 0..5 {
                for s in [-3, 0, 5, 14, 31] {
                    let sig = data(kind, 3000, 11 * taps as u64 + kind as u64);
                    let (mut p, mut v) = (Carried::new_ols(taps, k, bits), Carried::new_ols(taps, k, bits));
                    let (mut pi, mut vi) = (Carried::new_ols(taps, k, bits), Carried::new_ols(taps, k, bits));
                    for chunk in sig.chunks(700) {
                        let (mut a, mut b) = (chunk.to_vec(), chunk.to_vec());
                        p.run_impl::<false, false>(&mut a, s).unwrap();
                        unsafe { v.run_avx2::<false>(&mut b, s) }.unwrap();
                        assert_eq!(a, b, "forward differs (taps {taps} kind {kind} s {s})");
                        assert!(p.w == v.w && p.hist == v.hist && p.avg == v.avg, "state differs after forward (taps {taps} kind {kind} s {s})");
                        // decode the encoded chunk with the other implementation than the one that encoded it
                        let (mut c, mut d) = (a.clone(), a.clone());
                        pi.run_impl::<true, false>(&mut c, s).unwrap();
                        unsafe { vi.run_avx2::<true>(&mut d, s) }.unwrap();
                        assert_eq!(c, chunk, "portable inverse is not exact (taps {taps} kind {kind} s {s})");
                        assert_eq!(d, chunk, "avx2 inverse is not exact (taps {taps} kind {kind} s {s})");
                        assert!(pi.w == vi.w && pi.hist == vi.hist && pi.avg == vi.avg, "state differs after inverse");
                    }
                }
            }
        }
    }
}
