//! Hand-written SIMD kernels for the block-independent encoder's LPC analysis (residual search,
//!; autocorrelation) and the decoder's LPC reconstruction.
//! (The backward-adaptive mode's weight-update kernel went with that mode,.): "Maintain a portable scalar implementation as the
//! reference. SIMD implementations must produce identical output to the reference implementation" --
//! every kernel here is checked against its scalar twin by exhaustive-ish property tests below, not
//! assumed correct from reading the intrinsics. Runtime-dispatched (cached `is_x86_feature_detected!`
//! check,: never assume a feature is present just because the build machine has it -- the same
//! portability lesson /`cpufeatures.rs` already established for this project). x86_64 dispatch
//! order: AVX-512F or AVX2, whichever times faster on this CPU at first use (per kernel; 
//! measured AVX-512F faster on Sapphire Rapids, Zen 5 differs) > AVX2 > scalar; aarch64: NEON.

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::OnceLock;

/// `_mm256_mul_epi32` / `_mm512_mul_epi32` pinned to one `vpmuldq`. The intrinsics are lowered to
/// generic IR (sign-extend the low 32 bits of each lane, then a 64-bit multiply); in the unrolled
/// residual kernels LLVM hoists a broadcast coefficient's sign-extension out of the loop, loses
/// the pattern, and emits the 64x64 multiply emulation (three `vpmuludq`, shifts, adds) per tap --
/// measured 1.5-3x slower at orders below 16 on Zen 5 (2026-09-30).
/// Same result bit for bit; `pure, nomem` leaves LLVM free to schedule and hoist it.
/// Callers must have the matching target feature (avx2 / avx512f) enabled.
#[cfg(target_arch = "x86_64")]
macro_rules! vpmuldq256 { ($a:expr, $b:expr) => {{
    let r: std::arch::x86_64::__m256i;
    std::arch::asm!("vpmuldq {r}, {a}, {b}", r = lateout(ymm_reg) r, a = in(ymm_reg) $a, b = in(ymm_reg) $b,
        options(pure, nomem, nostack, preserves_flags));
    r
}} }
#[cfg(target_arch = "x86_64")]
macro_rules! vpmuldq512 { ($a:expr, $b:expr) => {{
    let r: std::arch::x86_64::__m512i;
    std::arch::asm!("vpmuldq {r}, {a}, {b}", r = lateout(zmm_reg) r, a = in(zmm_reg) $a, b = in(zmm_reg) $b,
        options(pure, nomem, nostack, preserves_flags));
    r
}} }

/// `step(j)` for `j` in `0..n`, fully unrolled in source: `n` is a const generic, so the guards fold
/// away. LLVM itself fully unrolls the tap loop only up to order 23 and keeps orders 24-32 as a loop
/// over the coefficients (reloaded from the stack each trip), ~1.6x slower per tap on Zen 5
/// (2026-09-30).
#[cfg(target_arch = "x86_64")]
macro_rules! taps32 {
    ($step:ident, $n:expr) => { taps32!(@ $step, $n; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31) };
    (@ $step:ident, $n:expr; $($j:literal)*) => { $( if $j < $n { $step($j); } )* };
}

/// AVX-512 dispatch gate, now per kernel. Each dispatch
/// site that has both an AVX-512F and an AVX2 kernel picks between them the first time it runs, by
/// timing the two on a fixed synthetic input (a few hundred microseconds, once per process) --
///  "do not assume AVX-512 is automatically faster": on Zen 5 (Ryzen AI 7 PRO 350) the
/// AVX-512F autocorrelation measured slower than the AVX2 one (2026-09-30),
/// while on Sapphire Rapids the AVX-512F kernels measured faster. The two kernels of
/// a site are bit-identical (differential tests below), so the choice changes speed only, never output.
/// `FAK_DISABLE_AVX512=1` forces AVX2 everywhere and `FAK_FORCE_AVX512=1` forces AVX-512F (both
/// skip the timing; for benchmarking). Read once, like the feature check.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq)]
enum Avx512Gate { Off, On, Timed }

#[cfg(target_arch = "x86_64")]
fn avx512_gate() -> Avx512Gate {
    static GATE: OnceLock<Avx512Gate> = OnceLock::new();
    *GATE.get_or_init(|| {
        let set = |k: &str| std::env::var_os(k).is_some_and(|v| v == "1");
        if !is_x86_feature_detected!("avx512f") || set("FAK_DISABLE_AVX512") { Avx512Gate::Off }
        else if set("FAK_FORCE_AVX512") || !avx2_available() { Avx512Gate::On }
        else { Avx512Gate::Timed }
    })
}

/// True if `a` ran faster than `b`: best of 9 timings each, the two alternated so that clock ramp-up
/// and interference hit both alike. Each call should take tens of microseconds, well above the timer
/// resolution.
#[cfg(target_arch = "x86_64")]
fn first_is_faster(a: impl Fn(), b: impl Fn()) -> bool {
    let time = |f: &dyn Fn()| { let t = std::time::Instant::now(); f(); t.elapsed() };
    let (mut ta, mut tb) = (std::time::Duration::MAX, std::time::Duration::MAX);
    for r in 0..9 {
        if r % 2 == 0 { ta = ta.min(time(&a)); tb = tb.min(time(&b)); } else { tb = tb.min(time(&b)); ta = ta.min(time(&a)); }
    }
    ta < tb
}

/// Deterministic 16-bit-range test signal for the timings (an LCG; content does not matter, only that
/// both kernels get the same work).
#[cfg(target_arch = "x86_64")]
fn timing_signal(n: usize) -> Vec<i64> {
    (0..n as u64).map(|i| (i.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48) as i64 - 32768).collect()
}

/// [`lpc_residuals`]' x86 kernel choice: AVX-512F if the gate allows and (when timed) it is faster.
#[cfg(target_arch = "x86_64")]
fn residuals_use_avx512() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| match avx512_gate() {
        Avx512Gate::Off => false,
        Avx512Gate::On => true,
        Avx512Gate::Timed => {
            let x = timing_signal(1024 + 32);
            let c: Vec<i64> = (0..32).map(|j| 1200 - 70 * j).collect();
            let (x, c) = (&x, &c);
            // A spread of orders, as the candidate search asks for.
            let run = |f: unsafe fn(&[i64], u32, &[i64]) -> Vec<i64>| move || for _ in 0..4 {
                for o in [2usize, 8, 16, 32] { std::hint::black_box(unsafe { f(&c[..o], 12, x) }); }
            };
            // Safety (both): features confirmed by the gate (avx512f) and `avx2_available`.
            first_is_faster(run(lpc_residuals_avx512_by_order), run(lpc_residuals_avx2_by_order))
        }
    })
}

/// [`autocorr`]'s x86 kernel choice for 33 lags, as [`residuals_use_avx512`].
#[cfg(target_arch = "x86_64")]
fn autocorr_use_avx512() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| match avx512_gate() {
        Avx512Gate::Off => false,
        Avx512Gate::On => true,
        Avx512Gate::Timed => {
            let w: Vec<f64> = timing_signal(4096).iter().map(|&x| x as f64).collect();
            let w = &w;
            let run = |f: unsafe fn(&[f64]) -> [f64; 33]| move || for _ in 0..2 { std::hint::black_box(unsafe { f(w) }); };
            // Safety (both): features confirmed by the gate (avx512f) and `avx2_available`.
            first_is_faster(run(autocorr33_avx512), run(autocorr33_avx2))
        }
    })
}

/// Whether AVX2 clones of plain loops elsewhere in the crate may run.
/// `FAK_DISABLE_AVX2_CLONES=1` forces their baseline builds (timing comparisons).
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| avx2_available() && std::env::var_os("FAK_DISABLE_AVX2_CLONES").is_none_or(|v| v != "1"))
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| is_x86_feature_detected!("avx2"))
}

/// LPC residuals, `out[i - order] = samples[i] - ((sum_j coeffs[j] * samples[i-1-j]) >> shift)` for
/// `i` in `order..samples.len()` -- the block-independent encoder's candidate search, ~61% of encode
/// instructions after scalar fix. Vectorized *across
/// output samples* (each coefficient broadcast once, multiplied against a shifted window of 4/8
/// consecutive samples), so there is no per-sample horizontal reduction -- the fixed per-call cost
/// that sank row-width-18 kernel doesn't arise.
///
/// Exactness: the SIMD kernels multiply with `mul_epi32` (signed 32x32 -> 64, exact), so they are
/// only entered when every sample and coefficient fits in `i32` -- checked here, O(n), not assumed;
/// otherwise the scalar reference runs. Real encoder input is <= 25 bits (24-bit PCM, +1 for the
/// stereo side channel) and coefficients at most `MAX_PRECISION` = 15 bits (`lpc.rs`), so the fast path is
/// the only one real audio takes. Accumulation is wrapping `i64` in every implementation (the
/// scalar reference uses explicit `wrapping_*`), so all paths agree bit-for-bit for *any* i32-range
/// input, including ones that would overflow -- real input can't (`lpc::residuals`' bound: <= 2^43).
pub fn lpc_residuals(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    lpc_residuals_with(coeffs, shift, samples, || fits_i32(samples))
}

/// Branchless (an OR-reduction vectorizes; a short-circuiting `all` does not): every x fits i32
/// iff x + 2^31 has nothing above bit 31.
fn fits_i32(v: &[i64]) -> bool { v.iter().fold(0u64, |a, &x| a | (x.wrapping_add(1 << 31) as u64 >> 32)) == 0 }

/// [`lpc_residuals`] with the samples' range check supplied by the caller (who may already know it).
#[cfg_attr(not(any(target_arch = "x86_64", target_arch = "aarch64")), allow(unused_variables))]
fn lpc_residuals_with(coeffs: &[i64], shift: u32, samples: &[i64], samples_fit_i32: impl FnOnce() -> bool) -> Vec<i64> {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        if shift < 64 && samples.len() > coeffs.len() && fits_i32(coeffs) && samples_fit_i32() {
            // NEON is mandatory on AArch64, so no runtime feature check (as `stage_weight_update`).
            #[cfg(target_arch = "aarch64")]
            return unsafe { lpc_residuals_neon_by_order(coeffs, shift, samples) };
            #[cfg(target_arch = "x86_64")]
            if residuals_use_avx512() {
                // Safety: feature confirmed by `is_x86_feature_detected!("avx512f")` (`avx512_gate`).
                return unsafe { lpc_residuals_avx512_by_order(coeffs, shift, samples) };
            }
            #[cfg(target_arch = "x86_64")]
            if avx2_available() {
                // Safety: feature confirmed by `is_x86_feature_detected!("avx2")`.
                return unsafe { lpc_residuals_avx2_by_order(coeffs, shift, samples) };
            }
        }
    }
    lpc_residuals_scalar(coeffs, shift, samples)
}

/// Which kernel [`lpc_residuals`] uses on this machine for in-range input (mirrors its dispatch;
/// on an AVX-512 CPU the first call runs the timing that picks it).
pub fn lpc_residuals_kernel() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if residuals_use_avx512() { return "avx512f"; }
        if avx2_available() { return "avx2"; }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    #[allow(unreachable_code)]
    "scalar"
}

/// Whether [`lpc_residuals_estimate`] uses the AVX-512 VNNI kernel for 16-bit input: the CPU has it,
/// the gate allows AVX-512, `FAK_DISABLE_VNNI=1` is not set, and (when timed) it beats the
/// exact kernel [`lpc_residuals`] picked.
#[cfg(target_arch = "x86_64")]
fn estimate_use_vnni() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        let have = is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni")
            && std::env::var_os("FAK_DISABLE_VNNI").is_none_or(|v| v != "1");
        match avx512_gate() {
            Avx512Gate::Off => false,
            _ if !have => false,
            Avx512Gate::On => true,
            Avx512Gate::Timed => {
                let x = timing_signal(1024 + 32);
                let c: Vec<i64> = (0..32).map(|j| 1200 - 70 * j).collect();
                let (x, c) = (&x, &c);
                let run = |vnni: bool| move || for _ in 0..4 {
                    for o in [8usize, 12, 16, 32] {
                        // Safety: features confirmed above; operands fit i16.
                        std::hint::black_box(if vnni { unsafe { lpc_residuals_vnni_by_order(&c[..o], 12, x) } } else { lpc_residuals(&c[..o], 12, x) });
                    }
                };
                first_is_faster(run(true), run(false))
            }
        }
    })
}

/// How the AVX-512 kernels are chosen on this machine, in words: off, forced, or timed against the
/// AVX2 kernels at first use.
pub fn avx512_policy() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512f") { return "not available (CPU has no AVX-512F)"; }
        return match avx512_gate() {
            Avx512Gate::Off => "disabled (FAK_DISABLE_AVX512=1)",
            Avx512Gate::On => "forced on (FAK_FORCE_AVX512=1)",
            Avx512Gate::Timed => "timed against AVX2 for each kernel at first use; the faster one is kept",
        };
    }
    #[allow(unreachable_code)]
    "not applicable"
}

/// The code path for the plain analysis loops (cross-channel sums, residual and Rice cost
/// estimates, long-term-prediction FFT) that exist as an AVX2 build and a baseline build.
pub fn analysis_loops_kernel() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    { return if avx2_enabled() { "avx2 builds" } else { "baseline (sse2) builds" }; }
    #[cfg(target_arch = "aarch64")]
    { return "neon (compiler-vectorised)"; }
    #[allow(unreachable_code)]
    "portable"
}

/// The decoder's LPC reconstruction in use: the blocked SIMD form or the scalar form.
pub fn reconstruct_kernel() -> &'static str {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        if !blocked_enabled() { return "scalar (FAK_DISABLE_BLOCKED=1)"; }
        #[cfg(target_arch = "x86_64")]
        { return if avx2_available() { "blocked avx2" } else { "blocked baseline" }; }
        #[cfg(target_arch = "aarch64")]
        { return "blocked neon"; }
    }
    #[allow(unreachable_code)]
    "scalar"
}

/// Makes every timed kernel choice now (see `avx512_gate`), if not yet made. The encoders call this
/// before starting their worker threads, so the timings run in a quiet process instead of beside
/// busy workers (under full load they picked the slower kernel in 4 of 16 trials on Zen 5).
pub fn choose_kernels() {
    #[cfg(target_arch = "x86_64")]
    { residuals_use_avx512(); autocorr_use_avx512(); estimate_use_vnni(); }
}

/// Which kernel [`lpc_residuals_estimate`] uses for 16-bit input at orders 8..=32 (as
/// [`lpc_residuals_kernel`]; elsewhere it is [`lpc_residuals`] or NEON's i32 form).
pub fn lpc_estimate_kernel() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if estimate_use_vnni() { return "avx512vnni"; }
    #[cfg(target_arch = "aarch64")]
    { return "neon-i32"; }
    #[allow(unreachable_code)]
    lpc_residuals_kernel()
}

/// Which kernel [`autocorr`] uses for 33 lags (every full block) on this machine, as
/// [`lpc_residuals_kernel`].
pub fn autocorr_kernel() -> &'static str {
    #[cfg(target_os = "macos")]
    if crate::accel::apple_enabled() { return "accelerate"; }
    #[cfg(target_arch = "x86_64")]
    {
        if autocorr_use_avx512() { return "avx512f"; }
        if avx2_available() { return "avx2"; }
    }
    "portable"
}

/// Portable reference: the plain per-sample dot product.
pub fn lpc_residuals_scalar(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    let order = coeffs.len();
    let n = samples.len();
    if n <= order { return Vec::new(); }
    (order..n).map(|i| {
        let mut acc = 0i64;
        for (j, &c) in coeffs.iter().enumerate() { acc = acc.wrapping_add(c.wrapping_mul(samples[i - 1 - j])); }
        samples[i].wrapping_sub(acc >> shift)
    }).collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpc_residuals_avx2(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    use std::arch::x86_64::*;
    let order = coeffs.len();
    let n = samples.len();
    // Every element is written below (vector part, then the scalar tail) before `set_len`; not
    // zero-filling first saved ~6% of encode instructions (profile).
    let mut out: Vec<i64> = Vec::with_capacity(n - order);
    let cv: Vec<__m256i> = coeffs.iter().map(|&c| _mm256_set1_epi64x(c)).collect();
    let cnt = _mm_cvtsi32_si128(shift as i32);
    let zero = _mm256_setzero_si256();
    let sp = samples.as_ptr();
    let mut i = order;
    while i + 4 <= n {
        let mut acc = zero;
        for (j, c) in cv.iter().enumerate() {
            // samples[i-1-j .. i+3-j]; i-1-j >= order-1-j >= 0, and i+3-j <= n-1.
            let s = _mm256_loadu_si256(sp.add(i - 1 - j) as *const __m256i);
            acc = _mm256_add_epi64(acc, vpmuldq256!(*c, s));
        }
        // Arithmetic >> for i64 lanes (AVX2 has none): ((x ^ m) >>> s) ^ m, m = sign mask.
        let m = _mm256_cmpgt_epi64(zero, acc);
        let pred = _mm256_xor_si256(_mm256_srl_epi64(_mm256_xor_si256(acc, m), cnt), m);
        let x = _mm256_loadu_si256(sp.add(i) as *const __m256i);
        _mm256_storeu_si256(out.as_mut_ptr().add(i - order) as *mut __m256i, _mm256_sub_epi64(x, pred));
        i += 4;
    }
    lpc_residuals_tail(coeffs, shift, samples, i, out.as_mut_ptr());
    // Safety: indices order..n were all written (vector loop up to `i`, tail from `i`).
    out.set_len(n - order);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn lpc_residuals_avx512(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    use std::arch::x86_64::*;
    let order = coeffs.len();
    let n = samples.len();
    // Every element is written below (vector part, then the scalar tail) before `set_len`; not
    // zero-filling first saved ~6% of encode instructions (profile).
    let mut out: Vec<i64> = Vec::with_capacity(n - order);
    let cv: Vec<__m512i> = coeffs.iter().map(|&c| _mm512_set1_epi64(c)).collect();
    let cnt = _mm_cvtsi32_si128(shift as i32);
    let sp = samples.as_ptr();
    let mut i = order;
    while i + 8 <= n {
        let mut acc = _mm512_setzero_si512();
        for (j, c) in cv.iter().enumerate() {
            let s = _mm512_loadu_si512(sp.add(i - 1 - j) as *const _);
            acc = _mm512_add_epi64(acc, vpmuldq512!(*c, s));
        }
        let pred = _mm512_sra_epi64(acc, cnt);
        let x = _mm512_loadu_si512(sp.add(i) as *const _);
        _mm512_storeu_si512(out.as_mut_ptr().add(i - order) as *mut _, _mm512_sub_epi64(x, pred));
        i += 8;
    }
    lpc_residuals_tail(coeffs, shift, samples, i, out.as_mut_ptr());
    // Safety: indices order..n were all written (vector loop up to `i`, tail from `i`).
    out.set_len(n - order);
    out
}

/// Order-specialised forms of the two kernels above:
/// one monomorphized copy per order 1..=32, so the coefficients are broadcast once into registers
/// and the dot product is fully unrolled, and two output vectors per iteration keep two
/// independent add chains in flight. Same arithmetic as the generic kernels (and the scalar
/// reference): `mul_epi32` products, wrapping `i64` sums in increasing `j`, arithmetic shift.
/// Orders outside 1..=32 fall back to the generic kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn lpc_residuals_avx512_by_order(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    macro_rules! by_order { ($($n:literal)*) => { match coeffs.len() {
        $($n => lpc_residuals_avx512_fixed::<$n>(coeffs, shift, samples),)*
        _ => lpc_residuals_avx512(coeffs, shift, samples),
    } } }
    by_order!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn lpc_residuals_avx512_fixed<const N: usize>(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    use std::arch::x86_64::*;
    let n = samples.len();
    let mut out: Vec<i64> = Vec::with_capacity(n - N);
    let mut cv = [_mm512_setzero_si512(); N];
    for j in 0..N { cv[j] = _mm512_set1_epi64(coeffs[j]); }
    let cnt = _mm_cvtsi32_si128(shift as i32);
    let sp = samples.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = N;
    while i + 16 <= n {
        let (mut a0, mut a1) = (_mm512_setzero_si512(), _mm512_setzero_si512());
        // samples[i-1-j .. i+15-j]: i-1-j >= N-1-j >= 0 and i+15-j <= n-1. Fully unrolled in
        // source (`taps32!`).
        let mut step = |j: usize| {
            let p = sp.add(i - 1 - j);
            a0 = _mm512_add_epi64(a0, vpmuldq512!(cv[j], _mm512_loadu_si512(p as *const _)));
            a1 = _mm512_add_epi64(a1, vpmuldq512!(cv[j], _mm512_loadu_si512(p.add(8) as *const _)));
        };
        taps32!(step, N);
        let x0 = _mm512_loadu_si512(sp.add(i) as *const _);
        let x1 = _mm512_loadu_si512(sp.add(i + 8) as *const _);
        _mm512_storeu_si512(op.add(i - N) as *mut _, _mm512_sub_epi64(x0, _mm512_sra_epi64(a0, cnt)));
        _mm512_storeu_si512(op.add(i - N + 8) as *mut _, _mm512_sub_epi64(x1, _mm512_sra_epi64(a1, cnt)));
        i += 16;
    }
    lpc_residuals_tail(coeffs, shift, samples, i, op);
    // Safety: indices N..n were all written (vector loop up to `i`, tail from `i`).
    out.set_len(n - N);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpc_residuals_avx2_by_order(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    macro_rules! by_order { ($($n:literal)*) => { match coeffs.len() {
        $($n => lpc_residuals_avx2_fixed::<$n>(coeffs, shift, samples),)*
        _ => lpc_residuals_avx2(coeffs, shift, samples),
    } } }
    by_order!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpc_residuals_avx2_fixed<const N: usize>(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    use std::arch::x86_64::*;
    let n = samples.len();
    let mut out: Vec<i64> = Vec::with_capacity(n - N);
    let mut cv = [_mm256_setzero_si256(); N];
    for j in 0..N { cv[j] = _mm256_set1_epi64x(coeffs[j]); }
    let cnt = _mm_cvtsi32_si128(shift as i32);
    let zero = _mm256_setzero_si256();
    let sra = |acc: __m256i| {
        // Arithmetic >> for i64 lanes (AVX2 has none): ((x ^ m) >>> s) ^ m, m = sign mask.
        let m = _mm256_cmpgt_epi64(zero, acc);
        _mm256_xor_si256(_mm256_srl_epi64(_mm256_xor_si256(acc, m), cnt), m)
    };
    let sp = samples.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = N;
    while i + 8 <= n {
        let (mut a0, mut a1) = (zero, zero);
        let mut step = |j: usize| {
            let p = sp.add(i - 1 - j);
            a0 = _mm256_add_epi64(a0, vpmuldq256!(cv[j], _mm256_loadu_si256(p as *const __m256i)));
            a1 = _mm256_add_epi64(a1, vpmuldq256!(cv[j], _mm256_loadu_si256(p.add(4) as *const __m256i)));
        };
        taps32!(step, N);
        let x0 = _mm256_loadu_si256(sp.add(i) as *const __m256i);
        let x1 = _mm256_loadu_si256(sp.add(i + 4) as *const __m256i);
        _mm256_storeu_si256(op.add(i - N) as *mut __m256i, _mm256_sub_epi64(x0, sra(a0)));
        _mm256_storeu_si256(op.add(i - N + 4) as *mut __m256i, _mm256_sub_epi64(x1, sra(a1)));
        i += 8;
    }
    lpc_residuals_tail(coeffs, shift, samples, i, op);
    // Safety: indices N..n were all written (vector loop up to `i`, tail from `i`).
    out.set_len(n - N);
    out
}

/// NEON form of [`lpc_residuals_avx2_by_order`]: one monomorphized kernel per order 1..=32.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lpc_residuals_neon_by_order(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    macro_rules! by_order { ($($n:literal)*) => { match coeffs.len() {
        $($n => lpc_residuals_neon_fixed::<$n>(coeffs, shift, samples),)*
        _ => lpc_residuals_neon_fixed::<0>(coeffs, shift, samples),
    } } }
    by_order!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
}

/// Same shape as the AVX2 kernel (vectorized across outputs, each coefficient broadcast once), on
/// the exact widening multiply-accumulate `smlal`/`smlal2` (`vmlal_s32`/`vmlal_high_s32`: 32x32 ->
/// 64 into two i64 lanes). Entered only when every operand fits `i32` (checked by the dispatcher),
/// so the samples are narrowed to `i32` once up front and loaded four at a time. `sshl` by
/// `-shift` is the arithmetic (flooring) shift of the scalar reference. `N == 0` means "order taken
/// from `coeffs.len()`" (only reached for orders above 32, which the encoder never asks for).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lpc_residuals_neon_fixed<const N: usize>(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    use std::arch::aarch64::*;
    let order = if N == 0 { coeffs.len() } else { N };
    let n = samples.len();
    let mut out: Vec<i64> = Vec::with_capacity(n - order);
    let s32: Vec<i32> = samples.iter().map(|&x| x as i32).collect();
    let c32: Vec<i32> = coeffs.iter().map(|&x| x as i32).collect();
    let neg = vdupq_n_s64(-(shift as i64));
    let sp = s32.as_ptr();
    let xp = samples.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = order;
    while i + 8 <= n {
        let (mut a0, mut a1, mut a2, mut a3) = (vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0));
        for j in 0..order {
            let c = vdupq_n_s32(*c32.get_unchecked(j));
            // samples[i-1-j .. i+7-j]; i-1-j >= 0 and i+7-j <= n-1.
            let p = sp.add(i - 1 - j);
            let s0 = vld1q_s32(p);
            let s1 = vld1q_s32(p.add(4));
            a0 = vmlal_s32(a0, vget_low_s32(c), vget_low_s32(s0));
            a1 = vmlal_high_s32(a1, c, s0);
            a2 = vmlal_s32(a2, vget_low_s32(c), vget_low_s32(s1));
            a3 = vmlal_high_s32(a3, c, s1);
        }
        let o = op.add(i - order);
        vst1q_s64(o, vsubq_s64(vld1q_s64(xp.add(i)), vshlq_s64(a0, neg)));
        vst1q_s64(o.add(2), vsubq_s64(vld1q_s64(xp.add(i + 2)), vshlq_s64(a1, neg)));
        vst1q_s64(o.add(4), vsubq_s64(vld1q_s64(xp.add(i + 4)), vshlq_s64(a2, neg)));
        vst1q_s64(o.add(6), vsubq_s64(vld1q_s64(xp.add(i + 6)), vshlq_s64(a3, neg)));
        i += 8;
    }
    lpc_residuals_tail(coeffs, shift, samples, i, op);
    // Safety: indices order..n were all written (vector loop up to `i`, tail from `i`).
    out.set_len(n - order);
    out
}

/// Residuals for *ranking* candidates by estimated cost, not for coding: on AArch64, when the
/// samples times `2^shift` stay well inside 32 bits, the prediction is accumulated in wrapping
/// `i32` (`mla`: four outputs per instruction, twice `lpc_residuals`' widening `smlal` rate) and
/// the result equals `lpc_residuals`' unless a prediction exceeds 2^31 (a wrapped one just looks
/// expensive, so the candidate loses the ranking). Everything else, and every other target, is
/// `lpc_residuals`. The residuals actually written to the stream always come from `lpc_residuals`.
pub fn lpc_residuals_estimate(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    #[cfg(target_arch = "aarch64")]
    {
        let fits = |v: &[i64]| v.iter().fold(0u64, |a, &x| a | (x.wrapping_add(1 << 31) as u64 >> 32)) == 0;
        if shift < 24 && samples.len() > coeffs.len() && fits(coeffs) {
            // OR of the magnitudes' bits (x ^ sign) bounds the peak within 2x, and vectorizes.
            let bits = samples.iter().fold(0u64, |a, &x| a | (x ^ (x >> 63)) as u64);
            if bits << shift < 1 << 30 {
                // Safety: NEON is part of the aarch64 baseline; operands checked to fit i32.
                return unsafe { lpc_residuals_neon_i32_by_order(coeffs, shift, samples) };
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    // From order 8: below it, packing the samples costs more than the cheaper taps save (256-sample
    // slices; from order 5 on 4096-sample blocks --, 2026-09-30).
    if (8..=32).contains(&coeffs.len()) && samples.len() > coeffs.len() && estimate_use_vnni() {
        // `vpdpwssd` takes i16 operands: every coefficient must fit, and every sample (the OR of the
        // magnitudes' bits, as above, < 2^15 means -2^15 <= x < 2^15).
        let bits = samples.iter().fold(0u64, |a, &x| a | (x ^ (x >> 63)) as u64);
        let c_fit = || coeffs.iter().fold(0u64, |a, &c| a | (c.wrapping_add(1 << 15) as u64 >> 16)) == 0;
        if shift < 24 && bits < 1 << 15 && bits << shift < 1 << 30 && c_fit() {
            // Safety: avx512f/bw/vnni confirmed by `estimate_use_vnni`; operands checked to fit i16.
            return unsafe { lpc_residuals_vnni_by_order(coeffs, shift, samples) };
        }
        // Wider samples (a loud side channel, 24-bit audio): the exact kernel, reusing the range
        // just computed (x fits i32 iff its magnitude bits are below 2^31) instead of a second pass.
        return lpc_residuals_with(coeffs, shift, samples, || bits < 1 << 31);
    }
    lpc_residuals(coeffs, shift, samples)
}

/// The x86 form of the estimate path above, on AVX-512 VNNI: `vpdpwssd` multiplies pairs of i16
/// lanes and adds both products into an i32 lane, so one instruction does 32 taps' worth of
/// multiply-adds (16 outputs x a coefficient pair) where `lpc_residuals`' `vpmuldq` does 8. The
/// samples are packed once per call as `p[k] = x[k] | x[k-1] << 16`, so a load at `p[i-1-2q]` holds
/// `(x[i-1-2q], x[i-2-2q])` in lane `i`, matching the coefficient pair `(c[2q], c[2q+1])` (an odd
/// order gets a zero last coefficient). Wrapping i32 accumulation, as the NEON path: equal to
/// `lpc_residuals` unless a prediction leaves i32 (2026-09-30).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn lpc_residuals_vnni_by_order(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    macro_rules! by_order { ($($n:literal)*) => { match coeffs.len() {
        $($n => lpc_residuals_vnni::<$n>(coeffs, shift, samples),)*
        _ => lpc_residuals(coeffs, shift, samples),
    } } }
    by_order!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn lpc_residuals_vnni<const N: usize>(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    use std::arch::x86_64::*;
    let n = samples.len();
    let mut out: Vec<i64> = Vec::with_capacity(n - N);
    let op = out.as_mut_ptr();
    // Pair array (p[0]'s high half is never read by a real tap).
    let mut pairs: Vec<i32> = Vec::with_capacity(n);
    pairs.push(samples[0] as u16 as i32);
    pairs.extend(samples.windows(2).map(|w| (w[1] as u16 as u32 | (w[0] as u16 as u32) << 16) as i32));
    let pp = pairs.as_ptr();
    let mut cv = [_mm512_setzero_si512(); 16];
    for q in 0..N.div_ceil(2) {
        let hi = if 2 * q + 1 < N { coeffs[2 * q + 1] as u16 as u32 } else { 0 };
        cv[q] = _mm512_set1_epi32((coeffs[2 * q] as u16 as u32 | hi << 16) as i32);
    }
    let cnt = _mm_cvtsi32_si128(shift as i32);
    // Lowest load: p[i-1-2q] at i = N, last q = (N-1)/2, is p[0] for odd N (whose high half, x[-1],
    // meets the zero pad coefficient) and p[1] for even N.
    let mut i = N;
    while i + 32 <= n {
        let (mut a0, mut a1) = (_mm512_setzero_si512(), _mm512_setzero_si512());
        let mut step = |q: usize| {
            let p = pp.add(i - 1 - 2 * q);
            a0 = _mm512_dpwssd_epi32(a0, _mm512_loadu_si512(p as *const _), cv[q]);
            a1 = _mm512_dpwssd_epi32(a1, _mm512_loadu_si512(p.add(16) as *const _), cv[q]);
        };
        taps32!(step, N.div_ceil(2));
        // x[i..] is the low half of p[i..], sign-extended.
        let x0 = _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(_mm512_loadu_si512(pp.add(i) as *const _)));
        let x1 = _mm512_srai_epi32::<16>(_mm512_slli_epi32::<16>(_mm512_loadu_si512(pp.add(i + 16) as *const _)));
        let r0 = _mm512_sub_epi32(x0, _mm512_sra_epi32(a0, cnt));
        let r1 = _mm512_sub_epi32(x1, _mm512_sra_epi32(a1, cnt));
        let o = op.add(i - N);
        _mm512_storeu_si512(o as *mut _, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(r0)));
        _mm512_storeu_si512(o.add(8) as *mut _, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(r0)));
        _mm512_storeu_si512(o.add(16) as *mut _, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(r1)));
        _mm512_storeu_si512(o.add(24) as *mut _, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(r1)));
        i += 32;
    }
    lpc_residuals_tail(coeffs, shift, samples, i, op);
    // Safety: indices N..n were all written (vector loop up to `i`, tail from `i`).
    out.set_len(n - N);
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lpc_residuals_neon_i32_by_order(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    macro_rules! by_order { ($($n:literal)*) => { match coeffs.len() {
        $($n => lpc_residuals_neon_i32::<$n>(coeffs, shift, samples),)*
        _ => lpc_residuals_neon_i32::<0>(coeffs, shift, samples),
    } } }
    by_order!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lpc_residuals_neon_i32<const N: usize>(coeffs: &[i64], shift: u32, samples: &[i64]) -> Vec<i64> {
    use std::arch::aarch64::*;
    let order = if N == 0 { coeffs.len() } else { N };
    let n = samples.len();
    let mut out: Vec<i64> = Vec::with_capacity(n - order);
    let s32: Vec<i32> = samples.iter().map(|&x| x as i32).collect();
    let c32: Vec<i32> = coeffs.iter().map(|&x| x as i32).collect();
    let neg = vdupq_n_s32(-(shift as i32));
    let sp = s32.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = order;
    while i + 8 <= n {
        let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
        for j in 0..order {
            let c = vdupq_n_s32(*c32.get_unchecked(j));
            let p = sp.add(i - 1 - j);
            a0 = vmlaq_s32(a0, c, vld1q_s32(p));
            a1 = vmlaq_s32(a1, c, vld1q_s32(p.add(4)));
        }
        let r0 = vsubq_s32(vld1q_s32(sp.add(i)), vshlq_s32(a0, neg));
        let r1 = vsubq_s32(vld1q_s32(sp.add(i + 4)), vshlq_s32(a1, neg));
        let o = op.add(i - order);
        vst1q_s64(o, vmovl_s32(vget_low_s32(r0)));
        vst1q_s64(o.add(2), vmovl_high_s32(r0));
        vst1q_s64(o.add(4), vmovl_s32(vget_low_s32(r1)));
        vst1q_s64(o.add(6), vmovl_high_s32(r1));
        i += 8;
    }
    lpc_residuals_tail(coeffs, shift, samples, i, op);
    out.set_len(n - order);
    out
}

/// Scalar remainder (< one vector of outputs) for the SIMD kernels, same arithmetic as the reference.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
/// Scalar tail of the SIMD kernels. Safety: `out` must be valid for writes at offsets
/// `from - order .. samples.len() - order`.
unsafe fn lpc_residuals_tail(coeffs: &[i64], shift: u32, samples: &[i64], from: usize, out: *mut i64) {
    let order = coeffs.len();
    for i in from..samples.len() {
        let mut acc = 0i64;
        for (j, &c) in coeffs.iter().enumerate() { acc = acc.wrapping_add(c.wrapping_mul(samples[i - 1 - j])); }
        out.add(i - order).write(samples[i].wrapping_sub(acc >> shift));
    }
}

/// Autocorrelation `r[lag] = sum_{i=lag}^{n-1} w[i] * w[i-lag]` for `lag` in `0..=max_lag` -- the LPC
/// analysis's other hot loop (21-26% of block-mode encode instructions).
///
/// Float, but still bit-identical to the reference: the reference sums each lag separately,
/// starting from `-0.0` (`f64: Sum`'s neutral element) and adding `w[i] * w[i-lag]` in increasing
/// `i`. The fast path vectorizes *across lags* instead of within one lag's sum, so every lag's
/// accumulator still sees exactly the same IEEE-754 operations in exactly the same order -- nothing
/// is reassociated, and rustc never contracts `a * b + c` into an FMA. The first `max_lag` samples,
/// where only some lags have a term, are done lag-by-lag in the same order before the all-lags loop.
///
/// Specialized for `max_lag == 32` (every full block: `lpc::MAX_ORDER`), where the 33 accumulators
/// live in registers (9 AVX2 / 5 AVX-512 vectors); other lags use the reference.
pub fn autocorr(w: &[f64], max_lag: usize) -> Vec<f64> {
    #[cfg(target_os = "macos")]
    if crate::accel::apple_enabled() { return autocorr_vdsp(w, max_lag); }
    if max_lag == 32 {
        #[cfg(target_arch = "x86_64")]
        {
            if autocorr_use_avx512() {
                // Safety: feature confirmed by `is_x86_feature_detected!("avx512f")` (`avx512_gate`).
                return unsafe { autocorr33_avx512(w) }.to_vec();
            }
            if avx2_available() {
                // Safety: feature confirmed by `is_x86_feature_detected!("avx2")`.
                return unsafe { autocorr33_avx2(w) }.to_vec();
            }
        }
        return autocorr_lanes::<33>(w).to_vec();
    }
    autocorr_scalar(w, max_lag)
}

/// `out[k] = sum_p sig[k + p] * filt[p]` on Accelerate's `vDSP_convD` (`sig.len() >= out.len() +
/// filt.len() - 1`). Its own summation order, so results differ from the sequential sums in the
/// last bits: only for the opt-in `--accel apple` (`accel::enable_apple`).
#[cfg(target_os = "macos")]
pub(crate) fn vdsp_correlate(sig: &[f64], filt: &[f64], out: &mut [f64]) {
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn vDSP_convD(a: *const f64, ia: isize, f: *const f64, i_f: isize, c: *mut f64, ic: isize, n: usize, p: usize);
    }
    if out.is_empty() { return; }
    if filt.is_empty() { out.fill(0.0); return; }
    assert!(sig.len() + 1 >= out.len() + filt.len());
    // Safety: the assert bounds every read (`k + p < out.len() + filt.len() - 1 <= sig.len()`),
    // and `out` has room for its `out.len()` results.
    unsafe { vDSP_convD(sig.as_ptr(), 1, filt.as_ptr(), 1, out.as_mut_ptr(), 1, out.len(), filt.len()) };
}

/// Autocorrelation on Accelerate ([`vdsp_correlate`]; opt-in `--accel apple`): ~3x faster than
/// [`autocorr`] on an Apple M4 (`examples/autocorr_accelerate.rs`) but **not** bit-identical.
#[cfg(target_os = "macos")]
pub fn autocorr_vdsp(w: &[f64], max_lag: usize) -> Vec<f64> {
    let n = w.len();
    let mut out = vec![0.0; max_lag + 1];
    if n == 0 { return out; }
    // r[lag] = sum_p padded[lag + p] * w[p]: the whole window as the filter, zero-padded signal.
    let mut padded = Vec::with_capacity(n + max_lag);
    padded.extend_from_slice(w);
    padded.resize(n + max_lag, 0.0);
    vdsp_correlate(&padded, w, &mut out);
    out
}

/// Portable reference: the per-lag sum `lpc.rs` had before.
pub fn autocorr_scalar(w: &[f64], max_lag: usize) -> Vec<f64> {
    let n = w.len();
    (0..=max_lag).map(|lag| (lag..n).map(|i| w[i] * w[i - lag]).sum()).collect()
}

/// All `L` lags (`max_lag = L - 1`) at once. `acc[m]` holds lag `L - 1 - m`, so that for sample `i`
/// the needed partners `w[i - lag]` are the contiguous, ascending window `w[i-max_lag..=i]` -- a
/// shape LLVM vectorizes without shuffles. Plain Rust: the target-feature wrappers below only let
/// LLVM use wider registers for the same element-wise operations.
#[inline(always)]
fn autocorr_lanes<const L: usize>(w: &[f64]) -> [f64; L] {
    let max_lag = L - 1;
    let n = w.len();
    let mut acc = [-0.0f64; L];
    for i in 0..max_lag.min(n) {
        for lag in 0..=i { acc[max_lag - lag] += w[i] * w[i - lag]; }
    }
    for i in max_lag..n {
        let x = w[i];
        let win: &[f64; L] = w[i - max_lag..=i].try_into().expect("window is exactly L long");
        for m in 0..L { acc[m] += x * win[m]; }
    }
    acc.reverse();
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn autocorr33_avx2(w: &[f64]) -> [f64; 33] { autocorr_lanes::<33>(w) }

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn autocorr33_avx512(w: &[f64]) -> [f64; 33] { autocorr_lanes::<33>(w) }

#[cfg(test)]
mod tests {
    #[test]
    fn estimate_residuals_equal_exact_when_predictions_stay_in_range() {
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        for trial in 0..400usize {
            let order = 1 + trial % 32;
            let n = order + 8 + (next(&mut seed) % 300) as usize;
            let shift = 8 + (trial % 7) as u32;
            let samples: Vec<i64> = (0..n).map(|_| (next(&mut seed) as i64).rem_euclid(1 << 14) - (1 << 13)).collect();
            let coeffs: Vec<i64> = (0..order).map(|_| ((next(&mut seed) as i64).rem_euclid(1 << 9)) - (1 << 8)).collect();
            assert_eq!(lpc_residuals_estimate(&coeffs, shift, &samples), lpc_residuals_scalar(&coeffs, shift, &samples), "order={order} n={n}");
        }
    }

    use super::*;

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// The AVX-512 VNNI estimate kernel against the scalar reference, where every prediction fits
    /// i32 (it is then exact): every order 1..=32, every length up to a few vectors past the order
    /// (all head/tail remainders), the full i16 sample range including both limits, and the
    /// wrap-free coefficient range `32 * 2^10 * 2^15 = 2^30`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vnni_estimate_matches_scalar_reference_in_range() {
        if !(is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni")) { return; }
        let mut seed = 0x0DDB_1A5E_5BAD_5EEDu64;
        for order in 1..=32usize {
            for n in order + 2..order + 100 {
                let samples: Vec<i64> = (0..n).map(|i| match (next(&mut seed) % 8, i % 3) {
                    (0, _) => -32768, (1, _) => 32767, _ => (next(&mut seed) as i64).rem_euclid(1 << 16) - (1 << 15),
                }).collect();
                let coeffs: Vec<i64> = (0..order).map(|_| (next(&mut seed) as i64).rem_euclid(1 << 11) - (1 << 10)).collect();
                let shift = (n % 16) as u32;
                // Safety: features checked above.
                let got = unsafe { lpc_residuals_vnni_by_order(&coeffs, shift, &samples) };
                assert_eq!(got, lpc_residuals_scalar(&coeffs, shift, &samples), "order={order} n={n} shift={shift}");
            }
        }
    }

    /// The timed per-kernel choice (`avx512_gate`) settles once and is what dispatch reports.
    #[test]
    fn kernel_choice_is_made_once_and_reported() {
        choose_kernels();
        let (r, a) = (lpc_residuals_kernel(), autocorr_kernel());
        assert!(["avx512f", "avx2", "neon", "scalar"].contains(&r), "{r}");
        assert!(["avx512f", "avx2", "accelerate", "portable"].contains(&a), "{a}");
        for _ in 0..3 {
            choose_kernels();
            assert_eq!((lpc_residuals_kernel(), autocorr_kernel()), (r, a));
        }
    }

    /// Differential test for `lpc_residuals`: every available kernel (AVX-512F, AVX2, dispatch)
    /// against the scalar reference, across every order 1..=32 (and 33, beyond MAX_ORDER), lengths
    /// spanning every tail remainder, every shift 0..=31, and operands at the full i32 limits --
    /// wider than real audio's 25-bit/14-bit ranges, which includes wrapping-overflow cases.
    #[test]
    fn lpc_residuals_kernels_match_scalar_reference() {
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        for trial in 0..1200usize {
            let order = 1 + trial % 33;
            let n = order + (next(&mut seed) % if trial % 4 == 0 { 300 } else { 70 }) as usize;
            let wide = trial % 3 == 0;
            let lim: i64 = if wide { i32::MAX as i64 } else { 1 << 24 };
            let samples: Vec<i64> = (0..n).map(|k| match (k + trial) % 11 {
                0 => lim, 1 => -lim - (wide as i64),
                _ => (next(&mut seed) as i64).rem_euclid(2 * lim) - lim,
            }).collect();
            let clim: i64 = if wide { i32::MAX as i64 } else { 1 << 13 };
            let coeffs: Vec<i64> = (0..order).map(|_| (next(&mut seed) as i64).rem_euclid(2 * clim) - clim).collect();
            let shift = (trial % 32) as u32;
            let want = lpc_residuals_scalar(&coeffs, shift, &samples);
            assert_eq!(lpc_residuals(&coeffs, shift, &samples), want, "dispatch order={order} n={n}");
            #[cfg(target_arch = "aarch64")]
            assert_eq!(unsafe { lpc_residuals_neon_by_order(&coeffs, shift, &samples) }, want, "neon order={order} n={n}");
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    assert_eq!(unsafe { lpc_residuals_avx2(&coeffs, shift, &samples) }, want, "avx2 order={order} n={n}");
                    assert_eq!(unsafe { lpc_residuals_avx2_by_order(&coeffs, shift, &samples) }, want, "avx2 by-order order={order} n={n}");
                }
                if is_x86_feature_detected!("avx512f") {
                    assert_eq!(unsafe { lpc_residuals_avx512(&coeffs, shift, &samples) }, want, "avx512 order={order} n={n}");
                    assert_eq!(unsafe { lpc_residuals_avx512_by_order(&coeffs, shift, &samples) }, want, "avx512 by-order order={order} n={n}");
                }
            }
        }
        // Out-of-i32-range operands must take the scalar path and still be right.
        let samples = vec![1i64 << 40, 3, -(1i64 << 35), 7, 9, 11, 13, 15, 17, 19];
        assert_eq!(lpc_residuals(&[2, -1], 3, &samples), lpc_residuals_scalar(&[2, -1], 3, &samples));
        assert_eq!(lpc_residuals(&[1i64 << 33], 0, &[1, 2, 3, 4, 5, 6, 7, 8, 9]), lpc_residuals_scalar(&[1i64 << 33], 0, &[1, 2, 3, 4, 5, 6, 7, 8, 9]));
        assert!(lpc_residuals(&[1, 2, 3], 0, &[1, 2, 3]).is_empty());
    }

    /// The blocked AVX2 / NEON reconstruction must match the scalar form exactly -- same samples, same
    /// consumed count -- for every order it handles, every length (full blocks and tails), full
    /// `i32`/`MAX_PRECISION` operand ranges, and residuals that push a sample out of `i32` at an
    /// arbitrary point (the hand-off to the caller's `i128` loop).
    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn lpc_reconstruct_blocked_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        if !is_x86_feature_detected!("avx2") { return; }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for trial in 0..3000usize {
            let order = 1 + trial % 32;
            let n = (next(&mut seed) % if trial % 5 == 0 { 400 } else { 40 }) as usize;
            let shift = (next(&mut seed) % 20) as u32;
            let cl = if trial % 3 == 0 { 1i64 << 15 } else { 1 << 10 };
            let rev: Vec<i32> = (0..order).map(|_| ((next(&mut seed) as i64).rem_euclid(2 * cl) - cl) as i32).collect();
            let warm: Vec<i32> = (0..order).map(|_| ((next(&mut seed) as i64).rem_euclid(1 << 24) - (1 << 23)) as i32).collect();
            let res: Vec<i64> = (0..n).map(|k| {
                let r = next(&mut seed);
                if trial % 7 == 0 && k == n / 2 { 1i64 << 40 } // forces the out-of-i32 hand-off
                else if r % 13 == 0 { (r as i64).rem_euclid(1 << 33) - (1 << 32) }
                else { (r as i64).rem_euclid(1 << 12) - (1 << 11) }
            }).collect();
            let (mut a, mut b) = (warm.clone(), warm.clone());
            let da = lpc_reconstruct_i32_scalar(&rev, shift, &res, &mut a);
            #[cfg(target_arch = "x86_64")]
            let db = unsafe { lpc_reconstruct_i32_avx2_by_order(&rev, shift, &res, &mut b) };
            #[cfg(target_arch = "aarch64")]
            let db = unsafe { lpc_reconstruct_i32_neon_by_order(&rev, shift, &res, &mut b) };
            assert_eq!((da, &a), (db, &b), "order={order} n={n} shift={shift}");
        }
    }

    /// The opt-in Accelerate correlation is not bit-identical (own summation order) but must agree
    /// with the sequential sums to rounding error, for every window length and lag count the
    /// encoder uses (and the empty/degenerate ones). Called directly, never through the global
    /// `--accel apple` flag, which would leak into the tests that demand bit-exactness.
    #[test]
    #[cfg(target_os = "macos")]
    fn vdsp_autocorr_and_correlate_match_the_reference_to_rounding() {
        let mut seed = 0xACCE_1E4A_7E00_0001u64;
        for n in [0usize, 1, 2, 5, 33, 100, 1024, 4097] {
            let w: Vec<f64> = (0..n).map(|_| (next(&mut seed) as i64 % 40001 - 20000) as f64 * 0.37).collect();
            for lag in [0usize, 1, 8, 32] {
                let (got, want) = (autocorr_vdsp(&w, lag), autocorr_scalar(&w, lag));
                assert_eq!(got.len(), lag + 1);
                let scale = want.first().copied().unwrap_or(0.0).abs().max(1.0);
                assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() <= 1e-12 * scale), "n={n} lag={lag}: {got:?} vs {want:?}");
            }
            for (filt_len, outs) in [(0usize, 3usize), (1, 1), (7, 5), (n.min(64), 9)] {
                if filt_len > n { continue; }
                let sig: Vec<f64> = (0..filt_len + outs - 1).map(|_| (next(&mut seed) as i64 % 2001 - 1000) as f64).collect();
                let filt: Vec<f64> = (0..filt_len).map(|_| (next(&mut seed) as i64 % 2001 - 1000) as f64).collect();
                let mut out = vec![7.0; outs];
                vdsp_correlate(&sig, &filt, &mut out);
                for (k, &o) in out.iter().enumerate() {
                    let want: f64 = (0..filt_len).map(|p| sig[k + p] * filt[p]).sum(); // exact: small integers
                    assert_eq!(o, want, "filt_len={filt_len} k={k}");
                }
            }
        }
    }

    /// `autocorr` must equal the per-lag reference *bit for bit* (signed zeros included) on every
    /// path: random windowed-audio-like data, blocks of all zeros / mixed signs / exact zeros in
    /// the middle, lengths below, at and above 33, and every `max_lag` 0..=33.
    #[test]
    fn autocorr_matches_scalar_reference_bitwise() {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let mut seed = 0x1357_9BDF_2468_ACE0u64;
        for trial in 0..200usize {
            let n = [0usize, 1, 2, 31, 32, 33, 34, 64, 257, 4096][trial % 10] + (trial / 10) % 3;
            let w: Vec<f64> = (0..n).map(|k| match (k + trial) % 13 {
                0 => 0.0, 1 => -0.0,
                _ => ((next(&mut seed) as i64 >> 40) as f64) * (1.0 + (k as f64) * 1e-3),
            }).collect();
            for max_lag in [0usize, 1, 7, 31, 32, 33] {
                if max_lag >= n.max(1) && max_lag != 32 { continue; }
                let want = autocorr_scalar(&w, max_lag);
                assert_eq!(bits(&autocorr(&w, max_lag)), bits(&want), "dispatch n={n} max_lag={max_lag}");
                if max_lag == 32 {
                    assert_eq!(bits(&autocorr_lanes::<33>(&w)), bits(&want), "portable n={n}");
                    #[cfg(target_arch = "x86_64")]
                    {
                        if is_x86_feature_detected!("avx2") {
                            assert_eq!(bits(&unsafe { autocorr33_avx2(&w) }), bits(&want), "avx2 n={n}");
                        }
                        if is_x86_feature_detected!("avx512f") {
                            assert_eq!(bits(&unsafe { autocorr33_avx512(&w) }), bits(&want), "avx512 n={n}");
                        }
                    }
                }
            }
        }
        // All-zero block: every lag must come out as the reference's exact signed zero.
        let z = vec![0.0f64; 100];
        assert_eq!(bits(&autocorr(&z, 32)), bits(&autocorr_scalar(&z, 32)));
    }
}

/// Fast path of `lpc::reconstruct_into` (decode side): LPC reconstruction with `i32` coefficients
/// and samples, `i64` accumulation. `buf` holds the `order = rev.len()` warmup samples followed by
/// the samples reconstructed so far; `rev` is the coefficients oldest-first. Appends one sample
/// per residual and returns how many residuals it consumed: all of them, unless a reconstructed
/// value doesn't fit `i32` -- then it stops *before* that sample and the caller finishes with its
/// `i128` reference loop (which also applies the hostile-stream bound). No legitimate stream gets
/// there: samples are at most 25 bits.
///
/// Exact, so identical to the `i128` loop: `|c| <= 2^15` (`lpc::MAX_PRECISION` = 16 bits), `|x| <
/// 2^31`, at most 32 terms -> `|acc| <= 2^51`; `|e| < 2^61` (`rice::decode`: quotient < 2^32, k <=
/// 30), so `e + (acc >> shift)` stays inside `i64`.
///
/// One monomorphized loop per order (1..=32), so each dot product is fully unrolled scalar code --
/// the per-order specialisation libFLAC also uses. Measured (in-process decode, 7 real files): 13-22% faster than a generic loop with an AVX2-vectorized dot
/// product, which was itself *slower* when compiled for the unrolled form (+16-35%): every sample
/// depends on the previous one, so a vector dot product pays a horizontal sum per sample on the
/// critical path. Plain portable code, no `target_feature`.
pub fn lpc_reconstruct_i32(rev: &[i32], shift: u32, res: &[i64], buf: &mut Vec<i32>) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if rev.len() >= BLOCKED_MIN_ORDER && blocked_enabled() && avx2_available() {
            // Safety: feature confirmed by `is_x86_feature_detected!("avx2")`. AVX2 on AVX-512
            // machines too: a 512-bit, 8-lane form measured 65-75% *slower* decode than scalar on
            // the test Xeon, this 4-lane one 9-18% faster.
            return unsafe { lpc_reconstruct_i32_avx2_by_order(rev, shift, res, buf) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if rev.len() >= BLOCKED_MIN_ORDER && blocked_enabled() {
            // NEON is mandatory on AArch64: no runtime feature check.
            return unsafe { lpc_reconstruct_i32_neon_by_order(rev, shift, res, buf) };
        }
    }
    lpc_reconstruct_i32_scalar(rev, shift, res, buf)
}

/// The portable form of [`lpc_reconstruct_i32`]: one unrolled scalar loop per order.
pub fn lpc_reconstruct_i32_scalar(rev: &[i32], shift: u32, res: &[i64], buf: &mut Vec<i32>) -> usize {
    macro_rules! by_order { ($($n:literal)*) => { match rev.len() { $($n => lpc_reconstruct_i32_fixed::<$n>(rev, shift, res, buf),)* _ => 0 } } }
    by_order!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
}

/// Shortest order the blocked kernel handles (below it the whole window is mostly inside a block,
/// so there is little to vectorize).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const BLOCKED_MIN_ORDER: usize = 4;

/// `FAK_DISABLE_BLOCKED=1` forces the scalar reconstruction (timing comparisons).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn blocked_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FAK_DISABLE_BLOCKED").is_none_or(|v| v != "1"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpc_reconstruct_i32_avx2_by_order(rev: &[i32], shift: u32, res: &[i64], buf: &mut Vec<i32>) -> usize {
    macro_rules! by_order { ($($n:literal)*) => { match rev.len() {
        $($n => lpc_reconstruct_i32_blocked4::<$n>(rev, shift, res, buf),)*
        _ => lpc_reconstruct_i32_scalar(rev, shift, res, buf),
    } } }
    by_order!(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
}

/// Blocked reconstruction Each prediction `sum_k rev[k] * buf[t-N+k]` for a block of 4 outputs at
/// `q..q+4` splits by how old the sample a term reads is:
///
/// * windows ending at least 3 blocks back (`<= q-9`, terms `k <= N-12`): AVX2 over all 4 lanes,
///   from memory (written two blocks ago or earlier, by then out of the store queue);
/// * the other terms that read `< q-4` in some lane (windows within `q-12..q-5`): AVX2 on windows
///   built in registers by `alignr` over the two blocks before the previous one, kept as vectors,
///   with zeros standing in for `>= q-4`, whose terms are the scalar part's;
/// * the previous block and the block itself (`q-4..q+3`): scalar, from registers, the newest
///   term added last.
///
/// So nothing just written is read back through memory: a 128-bit load of samples written by
/// 32-bit stores cannot be store-forwarded and waits for the stores to commit, which put the
/// earlier form's loads on the serial chain (~3.8 ns/sample at every order on a Ryzen 9 5900X),
/// and the serial chain from one sample to the next is one multiply, two adds and a shift.
///
/// Exact and identical to [`lpc_reconstruct_i32_fixed`]: the same `i32 x i32 -> i64` products,
/// summed in a different order, which integer addition does not notice (`|acc| <= 2^51`, no
/// overflow). Stops before the first reconstructed value that does not fit `i32`, returning how
/// many residuals it consumed, like the scalar form; the last partial block goes to the scalar form.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lpc_reconstruct_i32_blocked4<const N: usize>(rev: &[i32], shift: u32, res: &[i64], buf: &mut Vec<i32>) -> usize {
    use std::arch::x86_64::*;
    debug_assert!(N >= 4 && buf.len() >= N);
    let c: [i32; N] = rev.try_into().expect("order N");
    let c64: [i64; N] = std::array::from_fn(|k| c[k] as i64);
    let mut cv = [_mm256_setzero_si256(); N];
    for k in 0..N { cv[k] = _mm256_set1_epi32(c[k]); }
    // Terms k < km read only samples at least 3 blocks back: from memory.
    let km = N.saturating_sub(11);
    buf.reserve(res.len());
    let mut q = buf.len();
    let ptr = buf.as_mut_ptr();
    // Sample at `q + r` (r < 0), 0 before the history (never multiplied by a real coefficient).
    let hist = |q: usize, r: isize| if r >= -(N as isize) { *ptr.offset(q as isize + r) } else { 0 };
    let mut prev = [0i64; 4];
    for j in 0..4 { prev[j] = hist(q, j as isize - 4) as i64; }
    let mut p2 = _mm_setr_epi32(hist(q, -8), hist(q, -7), hist(q, -6), hist(q, -5));
    let mut p3 = _mm_setr_epi32(hist(q, -12), hist(q, -11), hist(q, -10), hist(q, -9));
    let zero = _mm_setzero_si128();
    // Samples q+s .. q+s+3 for s in -11..=-5, from [p3 | p2 | zeros].
    let window = |p3: __m128i, p2: __m128i, s: isize| match s + 12 {
        1 => _mm_alignr_epi8::<4>(p2, p3),
        2 => _mm_alignr_epi8::<8>(p2, p3),
        3 => _mm_alignr_epi8::<12>(p2, p3),
        4 => p2,
        5 => _mm_alignr_epi8::<4>(zero, p2),
        6 => _mm_alignr_epi8::<8>(zero, p2),
        7 => _mm_alignr_epi8::<12>(zero, p2),
        _ => unreachable!(),
    };
    let mut done = 0;
    while done + 4 <= res.len() {
        let (mut a0, mut a1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let base = ptr.add(q - N);
        let mut k = 0;
        while k + 2 <= km {
            a0 = _mm256_add_epi64(a0, _mm256_mul_epi32(cv[k], _mm256_cvtepi32_epi64(_mm_loadu_si128(base.add(k) as *const __m128i))));
            a1 = _mm256_add_epi64(a1, _mm256_mul_epi32(cv[k + 1], _mm256_cvtepi32_epi64(_mm_loadu_si128(base.add(k + 1) as *const __m128i))));
            k += 2;
        }
        if k < km { a0 = _mm256_add_epi64(a0, _mm256_mul_epi32(cv[k], _mm256_cvtepi32_epi64(_mm_loadu_si128(base.add(k) as *const __m128i)))); }
        for k in km..N.saturating_sub(4) {
            let w = window(p3, p2, k as isize - N as isize);
            a1 = _mm256_add_epi64(a1, _mm256_mul_epi32(cv[k], _mm256_cvtepi32_epi64(w)));
        }
        let mut outer = [0i64; 4];
        _mm256_storeu_si256(outer.as_mut_ptr() as *mut __m256i, _mm256_add_epi64(a0, a1));
        let mut xb = [0i64; 4];
        for l in 0..4 {
            let mut a = outer[l];
            // Samples q-4 .. q+l-1: k = N-4-l .. N-1, the newest last.
            for k in (N - 4).saturating_sub(l)..N {
                let r = k as isize + l as isize - N as isize;
                a += c64[k] * if r >= 0 { xb[r as usize] } else { prev[(r + 4) as usize] };
            }
            let x = res[done + l] + (a >> shift);
            if x != x as i32 as i64 { buf.set_len(q + l); return done + l; }
            xb[l] = x;
            *ptr.add(q + l) = x as i32;
        }
        p3 = p2;
        p2 = _mm_setr_epi32(prev[0] as i32, prev[1] as i32, prev[2] as i32, prev[3] as i32);
        prev = xb;
        q += 4;
        done += 4;
    }
    buf.set_len(q);
    done + lpc_reconstruct_i32_fixed::<N>(rev, shift, &res[done..], buf)
}


#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lpc_reconstruct_i32_neon_by_order(rev: &[i32], shift: u32, res: &[i64], buf: &mut Vec<i32>) -> usize {
    macro_rules! by_order { ($($n:literal)*) => { match rev.len() {
        $($n => lpc_reconstruct_i32_blocked4_neon::<$n>(rev, shift, res, buf),)*
        _ => lpc_reconstruct_i32_scalar(rev, shift, res, buf),
    } } }
    by_order!(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
}

/// NEON form of [`lpc_reconstruct_i32_blocked4`], same block structure and same exactness argument
/// (see there): `smlal`/`smlal2` for the exact `i32 x i32 -> i64` products of the four lanes,
/// `ext` where the AVX2 form uses `alignr` to build the windows just behind the previous block in
/// registers, so nothing just written is read back through memory. Returns like the scalar form.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lpc_reconstruct_i32_blocked4_neon<const N: usize>(rev: &[i32], shift: u32, res: &[i64], buf: &mut Vec<i32>) -> usize {
    use std::arch::aarch64::*;
    debug_assert!(N >= 4 && buf.len() >= N);
    let c: [i32; N] = rev.try_into().expect("order N");
    let c64: [i64; N] = std::array::from_fn(|k| c[k] as i64);
    let mut cv = [vdupq_n_s32(0); N];
    for k in 0..N { cv[k] = vdupq_n_s32(c[k]); }
    let km = N.saturating_sub(11);
    buf.reserve(res.len());
    let mut q = buf.len();
    let ptr = buf.as_mut_ptr();
    // Sample at `q + r` (r < 0), 0 before the history (never multiplied by a real coefficient).
    let hist = |q: usize, r: isize| if r >= -(N as isize) { *ptr.offset(q as isize + r) } else { 0 };
    let mut prev = [0i64; 4];
    for j in 0..4 { prev[j] = hist(q, j as isize - 4) as i64; }
    let load4 = |a: [i32; 4]| vld1q_s32(a.as_ptr());
    let mut p2 = load4([hist(q, -8), hist(q, -7), hist(q, -6), hist(q, -5)]);
    let mut p3 = load4([hist(q, -12), hist(q, -11), hist(q, -10), hist(q, -9)]);
    let zero = vdupq_n_s32(0);
    // Samples q+s .. q+s+3 for s in -11..=-5, from [p3 | p2 | zeros].
    let window = |p3: int32x4_t, p2: int32x4_t, s: isize| match s + 12 {
        1 => vextq_s32::<1>(p3, p2),
        2 => vextq_s32::<2>(p3, p2),
        3 => vextq_s32::<3>(p3, p2),
        4 => p2,
        5 => vextq_s32::<1>(p2, zero),
        6 => vextq_s32::<2>(p2, zero),
        7 => vextq_s32::<3>(p2, zero),
        _ => unreachable!(),
    };
    let mut done = 0;
    while done + 4 <= res.len() {
        // Two independent accumulator chains, each low/high pair of i64 lanes.
        let (mut l0, mut h0, mut l1, mut h1) = (vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0));
        let base = ptr.add(q - N);
        let mut k = 0;
        while k + 2 <= km {
            let w0 = vld1q_s32(base.add(k));
            let w1 = vld1q_s32(base.add(k + 1));
            l0 = vmlal_s32(l0, vget_low_s32(cv[k]), vget_low_s32(w0));
            h0 = vmlal_high_s32(h0, cv[k], w0);
            l1 = vmlal_s32(l1, vget_low_s32(cv[k + 1]), vget_low_s32(w1));
            h1 = vmlal_high_s32(h1, cv[k + 1], w1);
            k += 2;
        }
        if k < km {
            let w0 = vld1q_s32(base.add(k));
            l0 = vmlal_s32(l0, vget_low_s32(cv[k]), vget_low_s32(w0));
            h0 = vmlal_high_s32(h0, cv[k], w0);
        }
        for k in km..N.saturating_sub(4) {
            let w = window(p3, p2, k as isize - N as isize);
            l1 = vmlal_s32(l1, vget_low_s32(cv[k]), vget_low_s32(w));
            h1 = vmlal_high_s32(h1, cv[k], w);
        }
        let mut outer = [0i64; 4];
        vst1q_s64(outer.as_mut_ptr(), vaddq_s64(l0, l1));
        vst1q_s64(outer.as_mut_ptr().add(2), vaddq_s64(h0, h1));
        let mut xb = [0i64; 4];
        for l in 0..4 {
            let mut a = outer[l];
            // Samples q-4 .. q+l-1: k = N-4-l .. N-1, the newest last.
            for k in (N - 4).saturating_sub(l)..N {
                let r = k as isize + l as isize - N as isize;
                a += c64[k] * if r >= 0 { xb[r as usize] } else { prev[(r + 4) as usize] };
            }
            let x = res[done + l] + (a >> shift);
            if x != x as i32 as i64 { buf.set_len(q + l); return done + l; }
            xb[l] = x;
            *ptr.add(q + l) = x as i32;
        }
        p3 = p2;
        p2 = load4([prev[0] as i32, prev[1] as i32, prev[2] as i32, prev[3] as i32]);
        prev = xb;
        q += 4;
        done += 4;
    }
    buf.set_len(q);
    done + lpc_reconstruct_i32_fixed::<N>(rev, shift, &res[done..], buf)
}

#[inline(always)]
fn lpc_reconstruct_i32_fixed<const N: usize>(rev: &[i32], shift: u32, res: &[i64], buf: &mut Vec<i32>) -> usize {
    debug_assert!(buf.len() >= N && shift < 64);
    let c: [i32; N] = rev.try_into().expect("order N");
    let start = buf.len() - N;
    buf.reserve(res.len());
    for (i, &e) in res.iter().enumerate() {
        let win: &[i32; N] = buf[start + i..start + i + N].try_into().expect("window N");
        let mut acc = 0i64;
        for k in 0..N { acc += c[k] as i64 * win[k] as i64; }
        let x = e + (acc >> shift);
        if x != x as i32 as i64 { return i; }
        buf.push(x as i32);
    }
    res.len()
}
