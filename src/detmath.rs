//! Platform-independent replacements for the few libm functions the encoder's float analysis uses
//! (`cos` for LPC apodization windows, `log2` for LPC coefficient scaling, candidate ranking and
//! Rice-parameter estimation). `f64::cos`/`log2` call the *platform* math library, which is not
//! required to be correctly rounded: the first macOS CI run (Apple libm) produced different `.fak`
//! bytes than glibc (Linux x86_64 and ARM64) on 3/20 block-mode synthetic files, while all decoded
//! losslessly. Everything here uses only `+ - * /`, integer
//! bit manipulation and comparisons -- IEEE-754 requires those to be correctly rounded, and rustc
//! never fuses `a * b + c` into an FMA on its own -- so the results are bit-identical on every
//! target, whatever its libm. Encoder-only: the decoder is integer arithmetic throughout and never
//! calls these. Accuracy (~1e-15 relative) is far beyond what window shaping or a bit-cost estimate
//! needs; bit-reproducibility is the point.

use std::f64::consts::{FRAC_PI_2, LN_2};

/// `floor(log2(x))` for finite `x > 0`, computed exactly from the binary exponent -- no rounding at
/// all, so it is also *more* correct than `x.log2().floor()`, which can round `log2` of a value just
/// below a power of two up to the integer. Returns `i32::MIN` for `x <= 0` or non-finite `x`
/// (callers already guard those cases).
pub fn floor_log2(x: f64) -> i32 {
    if !(x > 0.0) || !x.is_finite() { return i32::MIN; }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        // Subnormal: value = mantissa * 2^-1074, so floor(log2) is the top set bit's position - 1074.
        let mant = bits & ((1u64 << 52) - 1);
        return 63 - mant.leading_zeros() as i32 - 1074;
    }
    exp - 1023
}

/// Natural log of `m` for `m` in `[sqrt(1/2), sqrt(2)]`: `2 * atanh(t)`, `t = (m-1)/(m+1)`, `|t| <=
/// 0.1716`, series to `t^23` (next term < 1e-19 relative). Fixed evaluation order.
fn ln_reduced(m: f64) -> f64 {
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    // Horner, highest-order term first: 1/1 + t2/3 + t2^2/5 + ... + t2^11/23.
    let mut s = 1.0 / 23.0;
    let mut k = 21.0;
    while k >= 1.0 {
        s = s * t2 + 1.0 / k;
        k -= 2.0;
    }
    2.0 * t * s
}

/// Deterministic `log2(x)`. `x == 0` -> `-inf`, `x < 0` or NaN -> NaN, `+inf` -> `+inf`, matching
/// `f64::log2`'s special cases.
pub fn log2(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 { return f64::NAN; }
    if x == 0.0 { return f64::NEG_INFINITY; }
    if x.is_infinite() { return f64::INFINITY; }
    // Normalize subnormals by an exact power-of-two scale first.
    let (x, bias) = if x < f64::MIN_POSITIVE { (x * (1u64 << 54) as f64, -54) } else { (x, 0) };
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023 + bias;
    // Mantissa in [1, 2): same significand, exponent field forced to 0.
    let mut m = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52));
    if m > std::f64::consts::SQRT_2 { m *= 0.5; e += 1; } // exact halving
    e as f64 + ln_reduced(m) / LN_2
}

/// `cos(r)` and `sin(r)` for `|r| <= pi/4`: Taylor series (sin to `r^17`, cos to `r^18`; next terms
/// < 1e-19), fixed Horner order.
fn cos_kernel(r: f64) -> f64 {
    let r2 = r * r;
    let mut s = 0.0;
    // cos r = sum_{k=0..9} (-1)^k r^(2k) / (2k)!, evaluated from the top.
    let mut k = 9.0;
    while k >= 1.0 {
        s = (1.0 - s) * r2 / ((2.0 * k) * (2.0 * k - 1.0));
        k -= 1.0;
    }
    1.0 - s
}
fn sin_kernel(r: f64) -> f64 {
    let r2 = r * r;
    let mut s = 0.0;
    let mut k = 8.0;
    while k >= 1.0 {
        s = (1.0 - s) * r2 / ((2.0 * k + 1.0) * (2.0 * k));
        k -= 1.0;
    }
    r * (1.0 - s)
}

/// Deterministic `cos(x)` for finite `x` (NaN for non-finite). Range reduction by `pi/2` with a
/// two-part constant (Cody-Waite): exact enough for the `|x| <= ~2*pi` arguments the window
/// functions use, and deterministic for any finite input.
pub fn cos(x: f64) -> f64 {
    if !x.is_finite() { return f64::NAN; }
    // pi/2 split so that k * PIO2_HI is exact for |k| < 2^20.
    const PIO2_HI: f64 = 1.570_796_326_734_125_614_166_259_765_625; // 0x3FF921FB54400000
    const PIO2_LO: f64 = 6.077_100_506_506_192_249_634_326_1e-11;  // pi/2 - PIO2_HI
    let k = (x / FRAC_PI_2).round();
    let r = (x - k * PIO2_HI) - k * PIO2_LO;
    match (k as i64).rem_euclid(4) {
        0 => cos_kernel(r),
        1 => -sin_kernel(r),
        2 => -cos_kernel(r),
        _ => sin_kernel(r),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_log2_is_exact_everywhere() {
        for e in -1074..1024 {
            let p = 2f64.powi(e);
            if p == 0.0 || !p.is_finite() { continue; }
            assert_eq!(floor_log2(p), e, "2^{e}");
            let below = f64::from_bits(p.to_bits() - 1);
            if below > 0.0 { assert_eq!(floor_log2(below), e - 1, "just below 2^{e}"); }
        }
        assert_eq!(floor_log2(3.0), 1);
        assert_eq!(floor_log2(0.75), -1);
        assert_eq!(floor_log2(f64::MAX), 1023);
        assert_eq!(floor_log2(0.0), i32::MIN);
        assert_eq!(floor_log2(-1.0), i32::MIN);
    }

    #[test]
    fn log2_matches_libm_closely_and_handles_special_cases() {
        let mut x = 1e-300;
        while x < 1e300 {
            let (d, l) = (log2(x), x.log2());
            assert!((d - l).abs() <= 1e-13 * l.abs().max(1.0), "x={x} det={d} libm={l}");
            x *= 1.37;
        }
        for e in -60..60 { assert_eq!(log2(2f64.powi(e)), e as f64, "exact at 2^{e}"); }
        assert!((log2(5e-324) - (-1074.0)).abs() < 1e-9, "smallest subnormal");
        assert_eq!(log2(0.0), f64::NEG_INFINITY);
        assert!(log2(-1.0).is_nan() && log2(f64::NAN).is_nan());
        assert_eq!(log2(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn cos_matches_libm_closely() {
        let mut worst = 0.0f64;
        for i in -20_000..=20_000 {
            let x = i as f64 * 0.001; // [-20, 20], well past the window functions' range
            worst = worst.max((cos(x) - x.cos()).abs());
        }
        assert!(worst < 1e-15, "max |det - libm| = {worst:e}");
        assert_eq!(cos(0.0), 1.0);
        assert!(cos(f64::INFINITY).is_nan() && cos(f64::NAN).is_nan());
    }

    /// Golden bit patterns: if any platform's build ever disagrees with these, the "same bytes on
    /// every target" guarantee this module exists for is broken there. Values computed by this
    /// implementation; the CI matrix runs this test on Linux/macOS/Windows x86_64 and ARM64.
    #[test]
    fn golden_bit_patterns_are_platform_independent() {
        let cases: [(f64, u64); 4] = [
            (cos(1.0), GOLDEN_COS_1),
            (cos(std::f64::consts::PI * 0.3), GOLDEN_COS_03PI),
            (log2(10.0), GOLDEN_LOG2_10),
            (log2(1234.5678), GOLDEN_LOG2_1234),
        ];
        for (i, (v, bits)) in cases.iter().enumerate() {
            assert_eq!(v.to_bits(), *bits, "case {i}: got {v:e} = {:#018x}", v.to_bits());
        }
    }
    const GOLDEN_COS_1: u64 = 0x3fe1_4a28_0fb5_068c;
    const GOLDEN_COS_03PI: u64 = 0x3fe2_cf23_0475_5a5f;
    const GOLDEN_LOG2_10: u64 = 0x400a_934f_0979_a371;
    const GOLDEN_LOG2_1234: u64 = 0x4024_8a21_f60f_faf4;
}
