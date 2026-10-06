//! Cross-channel prediction (format v12): a Fixed/LPC subframe may add a short FIR over an
//! *earlier* subframe of the same frame to its prediction. The FIR reads either that subframe's
//! decoded samples or its coded residual, at a window of lags around the current sample (the whole
//! reference frame is decoded first, so lags into its "future" are available):
//!
//! ```text
//! term[t] = (sum_{k<taps} c[k] * src[clamp(t + lag0 + k, 0, n-1)]) >> shift      (floor)
//! x[t]    = own_prediction(t) + term[t] + res[t]
//! ```
//!
//! The decoder adds `term` to the decoded residual and hands the result to the unchanged Fixed/LPC
//! reconstruction, so the recursive loop (and its SIMD paths) are untouched; the FIR itself is
//! non-recursive and cheap. Integer ranges: `|src| <= SOURCE_BOUND = 2^40` (checked; real sources are
//! at most ~2^27), `|c| < 2^15`, `taps <= 16`, so `|sum| < 2^40 * 2^15 * 2^4 = 2^59` fits `i64`.
use crate::bitio::{BitReader, BitReaderError, BitWriter};
use crate::lpc::{MAX_PRECISION, MIN_PRECISION};

pub const MAX_TAPS: usize = 16;
/// Stored as `lag0 + LAG_BIAS` in `LAG_BITS` bits, so `lag0` in `-32..=31`.
pub const LAG_BITS: u32 = 6;
pub const LAG_BIAS: i32 = 32;
pub const SOURCE_BOUND: i64 = 1 << 40;
pub const MAX_SHIFT: u32 = 31;

/// Which signal of the reference subframe the FIR reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source { Samples = 0, Residual = 1 }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrossParams {
    /// Index of the reference subframe within the frame (strictly earlier than the current one).
    pub ref_idx: u8,
    pub source: Source,
    pub lag0: i32,
    pub coeffs: Vec<i64>,
    pub shift: u32,
    pub precision: u32,
}

impl CrossParams {
    pub fn side_bits(&self) -> u64 { 8 + 1 + LAG_BITS as u64 + 4 + 4 + 5 + self.coeffs.len() as u64 * self.precision as u64 }

    pub fn write(&self, w: &mut BitWriter) {
        w.write_bits(self.ref_idx as u64, 8);
        w.write_bits(self.source as u64, 1);
        w.write_bits((self.lag0 + LAG_BIAS) as u64, LAG_BITS);
        w.write_bits((self.coeffs.len() - 1) as u64, 4);
        w.write_bits((self.precision - 1) as u64, 4);
        w.write_bits(self.shift as u64, 5);
        for &c in &self.coeffs { w.write_signed(c, self.precision); }
    }

    /// Reads and validates the fields; `n_refs` is how many earlier subframes the frame has.
    pub fn read(r: &mut BitReader, n_refs: usize) -> Result<Self, String> {
        let e = |e: BitReaderError| e.0.to_string();
        let ref_idx = r.read_bits(8).map_err(e)? as u8;
        if ref_idx as usize >= n_refs { return Err(format!("cross-channel reference {ref_idx} is not an earlier subframe")); }
        let source = if r.read_bits(1).map_err(e)? == 0 { Source::Samples } else { Source::Residual };
        let lag0 = r.read_bits(LAG_BITS).map_err(e)? as i32 - LAG_BIAS;
        let taps = r.read_bits(4).map_err(e)? as usize + 1;
        let precision = r.read_bits(4).map_err(e)? as u32 + 1;
        if !(MIN_PRECISION..=MAX_PRECISION).contains(&precision) { return Err(format!("invalid cross-channel precision {precision}")); }
        let shift = r.read_bits(5).map_err(e)? as u32;
        let mut coeffs = Vec::with_capacity(taps);
        for _ in 0..taps { coeffs.push(r.read_signed(precision).map_err(e)?); }
        Ok(CrossParams { ref_idx, source, lag0, coeffs, shift, precision })
    }
}

/// Scalar reference for one term.
pub fn term_ref(p: &CrossParams, src: &[i64], t: usize) -> i64 {
    let n = src.len() as i64;
    let mut acc = 0i64;
    for (k, &c) in p.coeffs.iter().enumerate() {
        let i = (t as i64 + p.lag0 as i64 + k as i64).clamp(0, n - 1) as usize;
        acc += c * src[i];
    }
    acc >> p.shift
}

/// Adds (`subtract == false`: the decoder) or subtracts (the encoder) `term(t)` to `res[i]` for
/// frame positions `t = start + i`. `src` is the whole reference frame. Fails if any source value
/// exceeds [`SOURCE_BOUND`]. The interior, where no index is clamped, runs a vector kernel when
/// available; the edges and everything else use [`term_ref`]'s arithmetic, and every path gives
/// identical results.
///
/// No overflow: `|term| < 2^59` (module docs) and a Rice-decoded residual is below `2^62` in
/// magnitude (`rice`: `k <= 30`, quotient `< 2^32`, escapes `<= 40` bits), so the sum stays inside
/// `i64`; an encoder-side residual is far smaller still.
pub fn apply(p: &CrossParams, src: &[i64], start: usize, res: &mut [i64], subtract: bool) -> Result<(), &'static str> {
    // Branch-free OR-reduction (vectorizes): any bit above the low 32 after biasing by 2^31 means
    // the value does not fit i32.
    let fits_i32 = src.iter().fold(0u64, |acc, &v| acc | ((v as u64).wrapping_add(1 << 31) >> 32)) == 0;
    if !fits_i32 && !source_in_bounds(src) { return Err("cross-channel source out of range (corrupted stream?)"); }
    let n = src.len() as i64;
    let taps = p.coeffs.len() as i64;
    let end = (start + res.len()) as i64;
    // Interior: t + lag0 >= 0 and t + lag0 + taps - 1 <= n - 1, clipped to start..end.
    let lo = (-(p.lag0 as i64)).clamp(start as i64, end);
    let hi = (n - (p.lag0 as i64 + taps - 1)).clamp(lo, end);
    for t in (start as i64..lo).chain(hi..end) {
        let v = term_ref(p, src, t as usize);
        let r = &mut res[t as usize - start];
        *r = if subtract { *r - v } else { *r + v };
    }
    if lo < hi {
        let (lo, hi) = (lo as usize, hi as usize);
        let base = (lo as i64 + p.lag0 as i64) as usize;
        let dst = &mut res[lo - start..hi - start];
        let window = &src[base..base + dst.len() + p.coeffs.len() - 1];
        #[cfg(target_arch = "x86_64")]
        {
            if fits_i32 && crate::simd::avx2_enabled() {
                // Safety: AVX2 confirmed; every operand fits i32 (coefficients by `precision <= 16`).
                unsafe { interior_avx2(&p.coeffs, p.shift, window, dst, subtract) };
                return Ok(());
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if fits_i32 {
                // Safety: NEON is AArch64 baseline (`simd.rs`); every operand fits i32 as above.
                unsafe { interior_neon(&p.coeffs, p.shift, window, dst, subtract) };
                return Ok(());
            }
        }
        interior_scalar(&p.coeffs, p.shift, window, dst, subtract);
    }
    Ok(())
}

/// `dst[i] +/-= (sum_k c[k] * window[i + k]) >> shift`: the scalar interior.
fn interior_scalar(c: &[i64], shift: u32, window: &[i64], dst: &mut [i64], subtract: bool) {
    for (i, d) in dst.iter_mut().enumerate() {
        let mut acc = 0i64;
        for (k, &ck) in c.iter().enumerate() { acc += ck * window[i + k]; }
        let t = acc >> shift;
        *d = if subtract { *d - t } else { *d + t };
    }
}

/// [`interior_scalar`] with `vpmuldq` (signed 32x32 -> 64-bit products, exact because every
/// operand fits `i32`), eight outputs per iteration. AVX2 has no 64-bit arithmetic shift, so
/// `x >> s` is computed as `((x ^ m) >>> s) ^ m` with `m` the sign mask (floor division, as
/// `>>` on `i64`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interior_avx2(c: &[i64], shift: u32, window: &[i64], dst: &mut [i64], subtract: bool) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let taps = c.len();
    let mut cv = [_mm256_setzero_si256(); MAX_TAPS];
    for (v, &x) in cv.iter_mut().zip(c) { *v = _mm256_set1_epi64x(x); }
    let cnt = _mm_cvtsi32_si128(shift as i32);
    let zero = _mm256_setzero_si256();
    let sra = |x: __m256i| -> __m256i {
        let m = _mm256_cmpgt_epi64(zero, x);
        _mm256_xor_si256(_mm256_srl_epi64(_mm256_xor_si256(x, m), cnt), m)
    };
    let wp = window.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0;
    while i + 8 <= n {
        let mut a0 = zero;
        let mut a1 = zero;
        for k in 0..taps {
            let x0 = _mm256_loadu_si256(wp.add(i + k) as *const __m256i);
            let x1 = _mm256_loadu_si256(wp.add(i + k + 4) as *const __m256i);
            a0 = _mm256_add_epi64(a0, _mm256_mul_epi32(x0, cv[k]));
            a1 = _mm256_add_epi64(a1, _mm256_mul_epi32(x1, cv[k]));
        }
        let (t0, t1) = (sra(a0), sra(a1));
        let r0 = _mm256_loadu_si256(dp.add(i) as *const __m256i);
        let r1 = _mm256_loadu_si256(dp.add(i + 4) as *const __m256i);
        let (o0, o1) = if subtract { (_mm256_sub_epi64(r0, t0), _mm256_sub_epi64(r1, t1)) } else { (_mm256_add_epi64(r0, t0), _mm256_add_epi64(r1, t1)) };
        _mm256_storeu_si256(dp.add(i) as *mut __m256i, o0);
        _mm256_storeu_si256(dp.add(i + 4) as *mut __m256i, o1);
        i += 8;
    }
    if i < n { interior_scalar(c, shift, &window[i..], &mut dst[i..], subtract); }
}

/// Outputs per stack-narrowed chunk in [`interior_neon`].
#[cfg(target_arch = "aarch64")]
const NEON_CHUNK: usize = 256;

/// [`interior_scalar`] with `smlal`/`smlal2` (signed 32x32 -> 64-bit multiply-accumulate, exact
/// because every operand fits `i32`), eight outputs per iteration. The window is narrowed to `i32`
/// once per chunk into a stack buffer, so each tap loads packed `i32` lanes instead of re-narrowing
/// `i64` loads. `sshl` by `-shift` is the arithmetic (floor) shift, as `>>` on `i64`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn interior_neon(c: &[i64], shift: u32, window: &[i64], dst: &mut [i64], subtract: bool) {
    use std::arch::aarch64::*;
    let taps = c.len();
    let mut cv = [vdupq_n_s32(0); MAX_TAPS];
    for (v, &x) in cv.iter_mut().zip(c) { *v = vdupq_n_s32(x as i32); }
    let sh = vdupq_n_s64(-(shift as i64));
    let mut narrow = [0i32; NEON_CHUNK + MAX_TAPS - 1];
    let mut done = 0;
    while dst.len() - done >= 8 {
        let m = (dst.len() - done).min(NEON_CHUNK) & !7;
        for (o, &x) in narrow.iter_mut().zip(&window[done..done + m + taps - 1]) { *o = x as i32; }
        let wp = narrow.as_ptr();
        let dp = dst.as_mut_ptr().add(done);
        let mut i = 0;
        while i < m {
            let (mut a0, mut a1, mut a2, mut a3) = (vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0));
            for k in 0..taps {
                let x0 = vld1q_s32(wp.add(i + k));
                let x1 = vld1q_s32(wp.add(i + k + 4));
                a0 = vmlal_s32(a0, vget_low_s32(x0), vget_low_s32(cv[k]));
                a1 = vmlal_high_s32(a1, x0, cv[k]);
                a2 = vmlal_s32(a2, vget_low_s32(x1), vget_low_s32(cv[k]));
                a3 = vmlal_high_s32(a3, x1, cv[k]);
            }
            for (j, a) in [a0, a1, a2, a3].into_iter().enumerate() {
                let t = vshlq_s64(a, sh);
                let r = vld1q_s64(dp.add(i + 2 * j));
                vst1q_s64(dp.add(i + 2 * j), if subtract { vsubq_s64(r, t) } else { vaddq_s64(r, t) });
            }
            i += 8;
        }
        done += m;
    }
    if done < dst.len() { interior_scalar(c, shift, &window[done..], &mut dst[done..], subtract); }
}

pub fn source_in_bounds(src: &[i64]) -> bool { src.iter().all(|&v| v.unsigned_abs() <= SOURCE_BOUND as u64) }

/// Normal equations for least squares over `t in t0..t1`: features `sigs[f.0][t + f.1]`, target
/// `sigs[target.0][t + target.1]`. Every index must be in range.
///
/// Every entry is a lagged cross-correlation `sum_s A[s] * B[s + delta]` over a range that differs
/// from entry to entry only at the edges. So per signal pair, one pass over a common range
/// computes the sums for the pair's whole span of `delta` at once (the inner loop runs across
/// `delta` and vectorizes, while each `delta`'s own sum stays sequential in `s`), and each entry
/// adds or removes only its few edge products. f64 throughout: products of samples up to 2^26 are
/// exact, and the rounded sums only feed a least-squares solve. The operation order is fixed (no
/// reassociation, no FMA contraction), so the result is identical on every target.
pub fn normal_equations(sigs: &[&[i64]], feats: &[(usize, isize)], target: (usize, isize), t0: usize, t1: usize) -> (Vec<f64>, Vec<f64>) {
    let d = feats.len();
    let f: Vec<Vec<f64>> = sigs.iter().map(|s| s.iter().map(|&v| v as f64).collect()).collect();
    // Canonical entry: (a, oa, b, ob) with (a, oa) <= (b, ob); delta = ob - oa; s ranges over
    // t0 + oa .. t1 + oa.
    let canon = |x: (usize, isize), y: (usize, isize)| if x <= y { (x, y) } else { (y, x) };
    let mut entries: Vec<((usize, isize), (usize, isize))> = Vec::with_capacity(d * (d + 1) / 2 + d);
    for i in 0..d {
        for j in i..d { entries.push(canon(feats[i], feats[j])); }
        entries.push(canon(feats[i], target));
    }
    // Per signal pair: delta span and the common s-range [s0, s1) valid for every entry's delta.
    struct Group { a: usize, b: usize, dmin: isize, dmax: isize, s0: isize, s1: isize, sums: Vec<f64> }
    let mut groups: Vec<Group> = Vec::new();
    for &((a, oa), (b, ob)) in &entries {
        let delta = ob - oa;
        let (x, y) = (t0 as isize + oa, t1 as isize + oa);
        match groups.iter_mut().find(|g| g.a == a && g.b == b) {
            Some(g) => { g.dmin = g.dmin.min(delta); g.dmax = g.dmax.max(delta); g.s0 = g.s0.max(x); g.s1 = g.s1.min(y); }
            None => groups.push(Group { a, b, dmin: delta, dmax: delta, s0: x, s1: y, sums: Vec::new() }),
        }
    }
    for g in groups.iter_mut() {
        // An empty common range (not seen in practice) leaves `sums` empty: entries then sum
        // their whole range directly.
        if g.s1 > g.s0 {
            let span = (g.dmax - g.dmin + 1) as usize;
            let mut acc = vec![0f64; span];
            lagged_sums(&f[g.a][g.s0 as usize..g.s1 as usize], &f[g.b][(g.s0 + g.dmin) as usize..], &mut acc);
            g.sums = acc;
        }
    }
    let entry = |(a, oa): (usize, isize), (b, ob): (usize, isize)| -> f64 {
        let g = groups.iter().find(|g| g.a == a && g.b == b).expect("group built above");
        let delta = ob - oa;
        let (fa, fb) = (&f[a], &f[b]);
        let dot = |from: isize, to: isize| -> f64 {
            let mut acc = 0f64;
            for s in from..to { acc += fa[s as usize] * fb[(s + delta) as usize]; }
            acc
        };
        let (x, y) = (t0 as isize + oa, t1 as isize + oa);
        if g.sums.is_empty() { return dot(x, y); }
        // [x, y) = [x, s0) + [s0, s1) + [s1, y); x <= s0 and s1 <= y by construction.
        dot(x, g.s0) + g.sums[(delta - g.dmin) as usize] + dot(g.s1, y)
    };
    let mut r = vec![0f64; d * d];
    let mut b = vec![0f64; d];
    let mut k = 0;
    for i in 0..d {
        for j in i..d {
            let (x, y) = entries[k];
            let v = entry(x, y);
            r[i * d + j] = v;
            r[j * d + i] = v;
            k += 1;
        }
        let (x, y) = entries[k];
        b[i] = entry(x, y);
        k += 1;
    }
    (r, b)
}

/// `acc[k] += a[s] * b[s + k]` for every `s < a.len()`, `k < acc.len()`, in increasing `s`
/// (`b.len() >= a.len() + acc.len() - 1`). Vectorized across `k` only, so every `acc[k]` sees the
/// same operations in the same order on every path.
fn lagged_sums(a: &[f64], b: &[f64], acc: &mut [f64]) {
    #[cfg(target_os = "macos")]
    if crate::accel::apple_enabled() {
        // Opt-in `--accel apple`: the same sums on Accelerate (different summation order).
        let mut t = vec![0f64; acc.len()];
        crate::simd::vdsp_correlate(&b[..a.len() + acc.len() - 1], a, &mut t);
        for (x, y) in acc.iter_mut().zip(&t) { *x += y; }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: feature confirmed by `is_x86_feature_detected!("avx2")`.
            return unsafe { lagged_sums_avx2(a, b, acc) };
        }
    }
    lagged_sums_body(a, b, acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lagged_sums_avx2(a: &[f64], b: &[f64], acc: &mut [f64]) { lagged_sums_body(a, b, acc) }

/// `W` lags held in registers across one pass over `s`, with `U` independent accumulator sets for
/// samples `s` in turn (`s % U`) so the adds do not wait on each other: `W * U` is 32 lags' worth,
/// eight vector chains. Each lag's sum is the fixed-order total `((t0 + t1) + t2) + ...`, so the result
/// is the same on every target (rounding differs from a single sequential sum, which no one relies on).
#[inline(always)]
fn lagged_block<const W: usize, const U: usize>(a: &[f64], b: &[f64], acc: &mut [f64]) {
    let n = a.len();
    let b = &b[..n + W - 1];
    let mut t = [[0f64; W]; U];
    let mut s = 0;
    while s + U <= n {
        for u in 0..U {
            let av = a[s + u];
            let row: &[f64; W] = b[s + u..s + u + W].try_into().unwrap();
            for l in 0..W { t[u][l] += av * row[l]; }
        }
        s += U;
    }
    while s < n {
        let av = a[s];
        let row: &[f64; W] = b[s..s + W].try_into().unwrap();
        for l in 0..W { t[0][l] += av * row[l]; }
        s += 1;
    }
    for l in 0..W {
        let mut v = t[0][l];
        for u in 1..U { v += t[u][l]; }
        acc[l] += v;
    }
}

#[inline(always)]
fn lagged_sums_body(a: &[f64], b: &[f64], acc: &mut [f64]) {
    let span = acc.len();
    let n = a.len();
    let mut k = 0;
    while k + 32 <= span { lagged_block::<32, 1>(a, &b[k..], &mut acc[k..k + 32]); k += 32; }
    if k + 16 <= span { lagged_block::<16, 2>(a, &b[k..], &mut acc[k..k + 16]); k += 16; }
    if k + 8 <= span { lagged_block::<8, 4>(a, &b[k..], &mut acc[k..k + 8]); k += 8; }
    if k + 4 <= span { lagged_block::<4, 8>(a, &b[k..], &mut acc[k..k + 4]); k += 4; }
    if k < span {
        // Fewer than four lags left: one more four-lag block, its extra lags dropped, when `b`
        // is long enough to read them; else lag by lag.
        if b.len() >= n + k + 3 {
            let mut tmp = [0f64; 4];
            lagged_block::<4, 8>(a, &b[k..], &mut tmp);
            for (x, y) in acc[k..].iter_mut().zip(tmp) { *x += y; }
        } else {
            while k < span { lagged_block::<1, 8>(a, &b[k..], &mut acc[k..k + 1]); k += 1; }
        }
    }
}

/// Solves `r a = b` (symmetric positive definite, row-major, both triangles filled) by Cholesky with
/// a tiny relative ridge. Only `+ - * / sqrt` in a fixed order: identical on every target.
pub fn solve_spd(mut r: Vec<f64>, b: &[f64]) -> Option<Vec<f64>> {
    let p = b.len();
    let mut tr = 0f64;
    for i in 0..p { tr += r[i * p + i]; }
    if !(tr > 0.0) || !tr.is_finite() { return None; }
    let ridge = tr / p as f64 * 1e-10;
    for i in 0..p { r[i * p + i] += ridge; }
    let mut l = vec![0f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = r[i * p + j];
            for k in 0..j { s -= l[i * p + k] * l[j * p + k]; }
            if i == j {
                if !(s > 0.0) { return None; }
                l[i * p + i] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }
    let mut z = vec![0f64; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i { s -= l[i * p + k] * z[k]; }
        z[i] = s / l[i * p + i];
    }
    let mut a = vec![0f64; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for k in i + 1..p { s -= l[k * p + i] * a[k]; }
        a[i] = s / l[i * p + i];
    }
    if a.iter().all(|v| v.is_finite()) { Some(a) } else { None }
}

/// Quantizes cross-channel taps to `precision` bits with the shift that fits the largest one.
pub fn quantize(a: &[f64], precision: u32) -> Option<(Vec<i64>, u32)> {
    let q = crate::lpc::quantize_at(a, precision)?;
    if q.shift > MAX_SHIFT { return None; }
    Some((q.coeffs, q.shift))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> i64 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (*seed >> 33) as i64 }

    #[test]
    fn terms_match_the_scalar_reference_everywhere() {
        let mut s = 7u64;
        for n in [1usize, 2, 5, 17, 100, 1000] {
            let src: Vec<i64> = (0..n).map(|_| lcg(&mut s) % (1 << 26) - (1 << 25)).collect();
            for lag0 in [-32, -5, -1, 0, 3, 31] {
                for taps in [1usize, 3, 11, 16] {
                    let coeffs: Vec<i64> = (0..taps).map(|_| lcg(&mut s) % (1 << 15) - (1 << 14)).collect();
                    let p = CrossParams { ref_idx: 0, source: Source::Samples, lag0, coeffs, shift: 13, precision: 16 };
                    for start in [0usize, n / 3] {
                        let base: Vec<i64> = (start..n).map(|_| lcg(&mut s) % 1000 - 500).collect();
                        for subtract in [false, true] {
                            let mut out = base.clone();
                            apply(&p, &src, start, &mut out, subtract).unwrap();
                            for (i, &o) in out.iter().enumerate() {
                                let t = term_ref(&p, &src, start + i);
                                assert_eq!(o, if subtract { base[i] - t } else { base[i] + t }, "n {n} lag0 {lag0} taps {taps} t {}", start + i);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The NEON interior against the scalar interior directly, across chunk boundaries
    /// (`NEON_CHUNK`), every tap count, shifts 0..=31 and i32-extreme operands.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_interior_matches_scalar() {
        let mut s = 11u64;
        for n in [0usize, 1, 7, 8, 9, 255, 256, 257, 263, 700, 1031] {
            for taps in 1..=MAX_TAPS {
                let window: Vec<i64> = (0..n + taps - 1).map(|_| match lcg(&mut s) % 7 {
                    0 => i32::MIN as i64,
                    1 => i32::MAX as i64,
                    _ => lcg(&mut s) % (1 << 27) - (1 << 26),
                }).collect();
                let c: Vec<i64> = (0..taps).map(|_| lcg(&mut s) % (1 << 16) - (1 << 15)).collect();
                let shift = (lcg(&mut s) % 32) as u32;
                let base: Vec<i64> = (0..n).map(|_| lcg(&mut s) % 1000 - 500).collect();
                for subtract in [false, true] {
                    let (mut a, mut b) = (base.clone(), base.clone());
                    interior_scalar(&c, shift, &window, &mut a, subtract);
                    unsafe { interior_neon(&c, shift, &window, &mut b, subtract) };
                    assert_eq!(a, b, "n {n} taps {taps} shift {shift} subtract {subtract}");
                }
            }
        }
    }

    #[test]
    fn terms_match_the_reference_with_sources_beyond_i32() {
        let mut s = 5u64;
        for bits in [20u32, 31, 32, 40] {
            let src: Vec<i64> = (0..300).map(|_| lcg(&mut s) % (1i64 << bits) - (1i64 << (bits - 1))).collect();
            let coeffs: Vec<i64> = (0..16).map(|_| lcg(&mut s) % (1 << 16) - (1 << 15)).collect();
            let p = CrossParams { ref_idx: 0, source: Source::Samples, lag0: -7, coeffs, shift: 15, precision: 16 };
            let mut out = vec![0i64; 300];
            apply(&p, &src, 0, &mut out, false).unwrap();
            for (t, &o) in out.iter().enumerate() { assert_eq!(o, term_ref(&p, &src, t), "bits {bits} t {t}"); }
        }
    }

    #[test]
    fn extreme_operands_do_not_overflow() {
        let src = vec![SOURCE_BOUND, -SOURCE_BOUND, SOURCE_BOUND, -SOURCE_BOUND];
        let p = CrossParams { ref_idx: 0, source: Source::Residual, lag0: -8, coeffs: vec![-(1 << 15); MAX_TAPS], shift: 0, precision: 16 };
        let mut out = vec![(1i64 << 62) - 1; 4];
        apply(&p, &src, 0, &mut out, false).unwrap();
        let mut out = vec![0i64; 4];
        assert!(apply(&p, &[SOURCE_BOUND + 1, 0, 0, 0], 0, &mut out, false).is_err());
    }

    #[test]
    fn params_roundtrip_and_reject_hostile_fields() {
        let p = CrossParams { ref_idx: 1, source: Source::Residual, lag0: -5, coeffs: vec![3, -7, 100, -4096], shift: 12, precision: 14 };
        let mut w = BitWriter::new();
        p.write(&mut w);
        let bytes = w.finish();
        assert_eq!(CrossParams::read(&mut BitReader::new(&bytes), 2).unwrap(), p);
        assert!(CrossParams::read(&mut BitReader::new(&bytes), 1).is_err(), "reference must be an earlier subframe");
        let mut w = BitWriter::new();
        w.write_bits(0, 8); w.write_bits(0, 1); w.write_bits(0, LAG_BITS); w.write_bits(0, 4); w.write_bits(0, 4);
        let bytes = w.finish();
        assert!(CrossParams::read(&mut BitReader::new(&bytes), 1).is_err(), "precision 1 is out of range");
    }

    #[test]
    fn normal_equations_match_the_direct_sums() {
        let mut s = 3u64;
        let a: Vec<i64> = (0..300).map(|_| lcg(&mut s) % (1 << 24) - (1 << 23)).collect();
        let b: Vec<i64> = (0..300).map(|_| lcg(&mut s) % (1 << 24) - (1 << 23)).collect();
        let feats = [(0usize, -1isize), (0, -2), (0, -3), (1, -2), (1, 0), (1, 1), (1, 2)];
        let (t0, t1) = (40usize, 280usize);
        let (r, rhs) = normal_equations(&[&a, &b], &feats, (0, 0), t0, t1);
        let sig = |f: (usize, isize), t: usize| -> i128 { [&a, &b][f.0][(t as isize + f.1) as usize] as i128 };
        for i in 0..feats.len() {
            for j in 0..feats.len() {
                let direct: i128 = (t0..t1).map(|t| sig(feats[i], t) * sig(feats[j], t)).sum();
                assert!((r[i * feats.len() + j] - direct as f64).abs() <= 1e-12 * (direct as f64).abs().max(1.0));
            }
            let direct: i128 = (t0..t1).map(|t| sig(feats[i], t) * sig((0, 0), t)).sum();
            assert!((rhs[i] - direct as f64).abs() <= 1e-12 * (direct as f64).abs().max(1.0));
        }
    }

    #[test]
    fn solve_recovers_a_known_mixture() {
        let mut s = 11u64;
        let a: Vec<i64> = (0..2000).map(|_| lcg(&mut s) % 20001 - 10000).collect();
        let y: Vec<i64> = (0..2000).map(|t| {
            let l = |o: isize| a[(t as isize + o).clamp(0, 1999) as usize];
            (3 * l(-1) + 5 * l(0) - 2 * l(2)) / 4
        }).collect();
        let feats = [(0usize, -1isize), (0, 0), (0, 1), (0, 2)];
        let (r, b) = normal_equations(&[&a, &y], &feats, (1, 0), 2, 1997);
        let c = solve_spd(r, &b).unwrap();
        for (got, want) in c.iter().zip([0.75, 1.25, 0.0, -0.5]) { assert!((got - want).abs() < 1e-3, "{c:?}"); }
    }
}
