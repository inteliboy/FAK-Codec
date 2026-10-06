//! Fixed integer predictors, orders 0-4 (the same classical differencing family FLAC uses).
//! All arithmetic is done in i64: worst case is order 4 with coefficients summing to
//! |4|+|6|+|4|+|1| = 15, so even at 25-bit samples (24-bit side channel) the accumulated
//! magnitude is far inside i64 range (explicit, checked integer bounds).

pub const MAX_ORDER: usize = 4;

/// Predict x[i] from the 1..=order previous samples (h[0] = x[i-1], h[1] = x[i-2], ...).
fn predict(order: u8, h: &[i64]) -> i64 {
    match order {
        0 => 0,
        1 => h[0],
        2 => 2 * h[0] - h[1],
        3 => 3 * h[0] - 3 * h[1] + h[2],
        4 => 4 * h[0] - 6 * h[1] + 4 * h[2] - h[3],
        _ => unreachable!("fixed predictor order must be 0..=4"),
    }
}

/// Residuals for samples[order..], given the full sample array (warmup = samples[..order]).
pub fn residuals(order: u8, samples: &[i64]) -> Vec<i64> {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: feature confirmed by `is_x86_feature_detected!("avx2")`.
            return unsafe { residuals_avx2(order, samples) };
        }
    }
    residuals_body(order, samples)
}

/// [`residuals`] compiled for AVX2: the same code, auto-vectorized.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn residuals_avx2(order: u8, samples: &[i64]) -> Vec<i64> { residuals_body(order, samples) }

/// Each order's fixed prediction written out over the input slice directly (the finite-difference
/// forms of `predict`), so every output is independent of the others and the loop vectorizes; the
/// same integer expressions as `predict` term for term (the history-shifting form is kept as the test reference).
#[inline(always)]
fn residuals_body(order: u8, x: &[i64]) -> Vec<i64> {
    let o = order as usize;
    let n = x.len();
    if n <= o { return Vec::new(); }
    match order {
        0 => x.to_vec(),
        1 => x.windows(2).map(|w| w[1] - w[0]).collect(),
        2 => x.windows(3).map(|w| w[2] - (2 * w[1] - w[0])).collect(),
        3 => x.windows(4).map(|w| w[3] - (3 * w[2] - 3 * w[1] + w[0])).collect(),
        4 => x.windows(5).map(|w| w[4] - (4 * w[3] - 6 * w[2] + 4 * w[1] - w[0])).collect(),
        _ => unreachable!("fixed predictor order must be 0..=4"),
    }
}

/// Sample magnitudes this codec ever legitimately produces stay well under this (25-bit samples
/// plus headroom). A reconstructed value beyond it can only come from a corrupted/hostile stream
/// (e.g. a maximal-width Rice escape residual): reject it immediately rather than letting it feed
/// back into the predictor, where repeated injection could otherwise compound across samples
/// (-- integer overflow / malformed-input robustness).
pub const SANE_SAMPLE_BOUND: i128 = 1 << 48;

/// Reconstruct the full sample sequence from warmup samples and residuals.
pub fn reconstruct(order: u8, warmup: &[i64], res: &[i64]) -> Result<Vec<i64>, &'static str> {
    let mut out = Vec::with_capacity(warmup.len() + res.len());
    out.extend_from_slice(warmup);
    reconstruct_into(order, warmup, res, &mut out)?;
    Ok(out)
}

/// [`reconstruct`] without the warmup: appends only the `res.len()` new samples to `out` (the
/// warmup being history already decoded -- format v9).
pub fn reconstruct_into(order: u8, warmup: &[i64], res: &[i64], out: &mut Vec<i64>) -> Result<(), &'static str> {
    let o = order as usize;
    debug_assert_eq!(warmup.len(), o);
    out.reserve(res.len());
    let mut h = [0i64; MAX_ORDER];
    for k in 0..o { h[k] = warmup[o - 1 - k]; }
    for &e in res {
        let x128 = e as i128 + predict(order, &h) as i128;
        if x128.abs() > SANE_SAMPLE_BOUND { return Err("reconstructed sample out of sane range (corrupted stream?)"); }
        let x = x128 as i64;
        out.push(x);
        for k in (1..o).rev() { h[k] = h[k - 1]; }
        if o > 0 { h[0] = x; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_check(order: u8, samples: &[i64]) {
        if samples.len() < order as usize { return; }
        let res = residuals(order, samples);
        let back = reconstruct(order, &samples[..order as usize], &res).unwrap();
        assert_eq!(back, samples, "order {order} mismatch");
    }

    /// The history-shifting form `residuals` had before, kept as the reference.
    fn residuals_reference(order: u8, samples: &[i64]) -> Vec<i64> {
        let o = order as usize;
        let n = samples.len();
        if n <= o { return Vec::new(); }
        let mut out = Vec::with_capacity(n - o);
        let mut h = [0i64; MAX_ORDER];
        for k in 0..o { h[k] = samples[o - 1 - k]; }
        for i in o..n {
            out.push(samples[i] - predict(order, &h));
            for k in (1..o).rev() { h[k] = h[k - 1]; }
            if o > 0 { h[0] = samples[i]; }
        }
        out
    }

    #[test]
    fn residuals_match_reference() {
        let mut s = 0x2545F4914F6CDD1Du64;
        for trial in 0..500usize {
            let n = trial % 90;
            let lim = if trial % 2 == 0 { 1i64 << 25 } else { 1 << 8 };
            let x: Vec<i64> = (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s as i64).rem_euclid(2 * lim) - lim }).collect();
            for order in 0..=MAX_ORDER as u8 {
                assert_eq!(residuals(order, &x), residuals_reference(order, &x), "order={order} n={n}");
                assert_eq!(residuals_body(order, &x), residuals_reference(order, &x));
            }
        }
    }

    #[test]
    fn all_orders_roundtrip_on_varied_signals() {
        let signals: Vec<Vec<i64>> = vec![
            vec![0; 10],
            (0..50).collect(),
            (0..50).map(|i| if i % 2 == 0 { i } else { -i }).collect(),
            vec![i32::MIN as i64, i32::MAX as i64, 0, i32::MIN as i64, i32::MAX as i64, -1, 1],
            vec![8_388_607, -8_388_608, 8_388_607, -8_388_608, 0, 0, 1, -1], // 24-bit extremes
            (0..200).map(|i: i64| ((i * 7919) % 4001) - 2000).collect(),
        ];
        for order in 0..=4u8 {
            for s in &signals { roundtrip_check(order, s); }
        }
    }

    #[test]
    fn empty_and_short_inputs_dont_panic() {
        for order in 0..=4u8 {
            assert!(residuals(order, &[]).is_empty());
            let short: Vec<i64> = (0..order as i64).collect();
            assert!(residuals(order, &short).is_empty());
        }
    }

    #[test]
    fn reconstruct_rejects_runaway_hostile_residuals() {
        // A single huge residual (as could come from a maximal Rice escape) must be rejected,
        // not silently overflow or panic while it feeds back into the predictor.
        let huge = 1i64 << 60;
        assert!(reconstruct(4, &[0, 0, 0, 0], &[huge, huge, huge, huge, huge]).is_err());
    }
}
