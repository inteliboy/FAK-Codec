//! Linear predictive coding: windowed-autocorrelation + Levinson-Durbin to estimate float
//! coefficients, then quantized to fixed-point integers (with error feedback) so encoder and
//! decoder compute bit-identical integer predictions. This is the higher-order/adaptive
//! predictor.1 asks for, alongside the fixed predictors in `predictors.rs`.
use crate::detmath;
use crate::predictors::SANE_SAMPLE_BOUND;

pub const MAX_ORDER: usize = 32;
/// Default coefficient precision (bits, sign included): what the order search quantizes at before
/// the encoder refines the winner's precision per subframe (format v9).
pub const PRECISION: u32 = 14;
/// Range of per-subframe coefficient precisions the format allows. The upper bound is what the
/// integer-range argument in [`residuals`] assumes.
pub const MIN_PRECISION: u32 = 3;
pub const MAX_PRECISION: u32 = 16;
pub const MAX_SHIFT: u32 = 31;

/// Apodization window shapes tried before autocorrelation. A single window is a systematically
/// worse fit for some real content: FLAC's own higher presets (e.g. `-8`'s `subdivide_tukey(5)`)
/// already try several per block for exactly this reason. Measured here (`examples/`
/// `window_function_probe.rs`, real production `rice::cost_bits` on real `lpc::residuals`, 6
/// genre-diverse real corpus files): trying all four and keeping whichever real-cost-wins per block
/// gave a real, consistent win on every file (-0.02% to -0.65%, aggregate -0.33%), including the
/// project's one known FLAC-parity outlier (a dense orchestral recording, -0.65%, the largest single
/// -file win in the set) -- not an orchestral-only effect, a broad one. Encoder-only cost (the
/// decoder only ever sees the resulting quantized integer coefficients, identical either way).
#[derive(Clone, Copy)]
enum Window { Welch, Tukey50, Hann, Rectangular }
/// The windows the analysis tries. Welch and Rectangular were dropped after they stopped paying:
/// on 9 real excerpts (16/24-bit, 44.1-96 kHz) the four-window set and {Welch, Tukey, Hann} gave
/// byte-identical files at `normal` and `max` (Rectangular never won), and {Tukey, Hann} was
/// +0.002% at `fast`, `normal` and `max` for 8-20% less encode time. `FAK_WINDOWS=all` restores the four for testing.
const WINDOWS: &[Window] = &[Window::Tukey50, Window::Hann];
const ALL_WINDOWS: &[Window] = &[Window::Welch, Window::Tukey50, Window::Hann, Window::Rectangular];

fn active_windows() -> &'static [Window] {
    static ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ALL.get_or_init(|| matches!(std::env::var("FAK_WINDOWS").as_deref(), Ok("all"))) { ALL_WINDOWS } else { WINDOWS }
}

fn window_fn(kind: Window, n: usize, i: usize) -> f64 {
    if n <= 1 { return 1.0; }
    let nm1 = (n - 1) as f64;
    match kind {
        Window::Welch => {
            let t = (i as f64 - nm1 / 2.0) / (nm1 / 2.0);
            1.0 - t * t
        }
        Window::Hann => 0.5 - 0.5 * detmath::cos(2.0 * std::f64::consts::PI * i as f64 / nm1),
        Window::Tukey50 => {
            // Flat middle 50%, cosine-tapered 25% on each side -- FLAC's own `-8` base window.
            let alpha = 0.5;
            let edge = (alpha * nm1 / 2.0).floor();
            let x = i as f64;
            if x < edge {
                0.5 * (1.0 + detmath::cos(std::f64::consts::PI * (x / edge - 1.0)))
            } else if x > nm1 - edge {
                0.5 * (1.0 + detmath::cos(std::f64::consts::PI * ((x - (nm1 - edge)) / edge)))
            } else {
                1.0
            }
        }
        Window::Rectangular => 1.0,
    }
}

thread_local! {
    /// Window coefficients per block length, one table per `WINDOWS` entry. They depend only on
    /// `(kind, n)`, yet were recomputed for every channel of every block (four channels -- L/R/M/S
    /// -- per stereo block, with a `detmath::cos` per sample for two of the four windows): ~6-11% of
    /// block-mode encode instructions (profile). Variable block size analyses
    /// up to five lengths per chunk plus a chunk's shorter tail, so the cache keeps the last
    /// `WINDOW_CACHE_LEN` lengths (a single-entry cache thrashed: ~4% of encode).
    /// Same values, computed by the same `window_fn`, so output is unchanged.
    static WINDOW_CACHE: std::cell::RefCell<Vec<(usize, Vec<Vec<f64>>)>> = const { std::cell::RefCell::new(Vec::new()) };
}

const WINDOW_CACHE_LEN: usize = 8;

fn windowed(samples: &[i64], kind: Window) -> Vec<f64> {
    let n = samples.len();
    if n <= 1 { return samples.iter().map(|&s| s as f64).collect(); }
    WINDOW_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let idx = match cache.iter().position(|(len, _)| *len == n) {
            Some(i) => i,
            None => {
                if cache.len() == WINDOW_CACHE_LEN { cache.remove(0); }
                cache.push((n, ALL_WINDOWS.iter().map(|&k| (0..n).map(|i| window_fn(k, n, i)).collect()).collect()));
                cache.len() - 1
            }
        };
        let win = &cache[idx].1[kind as usize];
        samples.iter().zip(win).map(|(&s, &c)| s as f64 * c).collect()
    })
}

/// Implemented in `simd::autocorr` (vectorized across lags, bit-identical to the per-lag sum).
fn autocorr(w: &[f64], max_lag: usize) -> Vec<f64> { crate::simd::autocorr(w, max_lag) }

/// Levinson-Durbin recursion. Returns, for every order 1..=k it successfully reached, the
/// coefficients (`coeffs[order-1]` has `order` entries) *and* the recursion's own predicted
/// residual variance at that order (`err[order-1]`) -- a free byproduct used to rank orders
/// without materializing residuals (see `candidates`). Stops early if the recursion becomes
/// numerically unstable (near-zero error), which happens on near-silent or exactly periodic blocks.
fn levinson_all_orders(r: &[f64], max_order: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut coeffs_out = Vec::with_capacity(max_order);
    let mut err_out = Vec::with_capacity(max_order);
    let mut a = vec![0f64; max_order + 1];
    let mut err = r[0];
    if err <= 0.0 { return (coeffs_out, err_out); }
    let mut tmp = vec![0f64; max_order + 1];
    for i in 1..=max_order {
        if err < 1e-9 * r[0].max(1.0) { break; }
        let mut acc = r[i];
        for j in 1..i { acc -= a[j] * r[i - j]; }
        let k = acc / err;
        if !k.is_finite() || k.abs() >= 1.0 { break; }
        tmp[..i].copy_from_slice(&a[..i]);
        a[i] = k;
        for j in 1..i { a[j] = tmp[j] - k * tmp[i - j]; }
        err *= 1.0 - k * k;
        coeffs_out.push(a[1..=i].to_vec());
        err_out.push(err);
        if err <= 0.0 { break; }
    }
    (coeffs_out, err_out)
}

/// [`levinson_all_orders`]'s error sequence alone (the same recursion, so the same values), for
/// callers that rank orders without needing the coefficients: no per-order allocation.
fn levinson_errors(r: &[f64], max_order: usize) -> Vec<f64> {
    let mut err_out = Vec::with_capacity(max_order);
    let mut a = [0f64; MAX_ORDER + 1];
    let mut tmp = [0f64; MAX_ORDER + 1];
    let mut err = r[0];
    if err <= 0.0 { return err_out; }
    for i in 1..=max_order {
        if err < 1e-9 * r[0].max(1.0) { break; }
        let mut acc = r[i];
        for j in 1..i { acc -= a[j] * r[i - j]; }
        let k = acc / err;
        if !k.is_finite() || k.abs() >= 1.0 { break; }
        tmp[..i].copy_from_slice(&a[..i]);
        a[i] = k;
        for j in 1..i { a[j] = tmp[j] - k * tmp[i - j]; }
        err *= 1.0 - k * k;
        err_out.push(err);
        if err <= 0.0 { break; }
    }
    err_out
}

#[derive(Clone)]
pub struct QuantizedLpc { pub coeffs: Vec<i64>, pub shift: u32, pub precision: u32 }

/// Coefficients `0..COEFF_SPLIT` (the ones on the most recent samples, typically the largest) and
/// the rest each get their own Rice parameter (format v15).
const COEFF_SPLIT: usize = 2;

fn coeff_groups(c: &[i64]) -> (&[i64], &[i64]) { c.split_at(COEFF_SPLIT.min(c.len())) }

fn coeff_group_best(g: &[i64]) -> (u32, u64) {
    (0..16u32).map(|k| (k, g.iter().map(|&c| (crate::rice::zigzag(c) >> k) + 1 + k as u64).sum::<u64>()))
        .min_by_key(|&(_, b)| b).unwrap()
}

/// Exact bits [`write_coeffs`] spends on `coeffs`: per group a 4-bit Rice parameter (none for an
/// empty second group) and the Rice codes of the zigzag values. Speech codecs likewise entropy-code
/// their predictor parameters rather than storing them at a fixed width.
pub fn coeff_bits(coeffs: &[i64]) -> u64 {
    let (a, b) = coeff_groups(coeffs);
    4 + coeff_group_best(a).1 + if b.is_empty() { 0 } else { 4 + coeff_group_best(b).1 }
}

pub fn write_coeffs(w: &mut crate::bitio::BitWriter, coeffs: &[i64]) {
    let (a, b) = coeff_groups(coeffs);
    for g in [a, b] {
        if g.is_empty() { continue; }
        let k = coeff_group_best(g).0;
        w.write_bits(k as u64, 4);
        for &c in g {
            let z = crate::rice::zigzag(c);
            w.write_unary(z >> k);
            if k > 0 { w.write_bits(z & ((1 << k) - 1), k); }
        }
    }
}

/// Reads `order` coefficients written by [`write_coeffs`], rejecting any that does not fit a
/// `precision`-bit signed value (the bound the reconstruction's overflow analysis assumes).
pub fn read_coeffs(r: &mut crate::bitio::BitReader, order: usize, precision: u32) -> Result<Vec<i64>, &'static str> {
    let mut out = Vec::with_capacity(order);
    let lim = 1i64 << (precision - 1);
    for len in [COEFF_SPLIT.min(order), order.saturating_sub(COEFF_SPLIT)] {
        if len == 0 { continue; }
        let k = r.read_bits(4).map_err(|e| e.0)? as u32;
        for _ in 0..len {
            let c = crate::rice::unzigzag(r.read_rice(k, 1 << 17).map_err(|e| e.0)?);
            if !(-lim..lim).contains(&c) { return Err("LPC coefficient exceeds its declared precision (corrupted stream?)"); }
            out.push(c);
        }
    }
    Ok(out)
}

/// Quantize float LPC coefficients to `PRECISION`-bit signed fixed-point, with error feedback
/// (each coefficient's rounding error is carried into the next) to keep quantization noise low.
pub fn quantize(a: &[f64]) -> Option<QuantizedLpc> { quantize_at(a, PRECISION) }

/// [`quantize`] at an explicit `precision` in `MIN_PRECISION..=MAX_PRECISION`.
pub fn quantize_at(a: &[f64], precision: u32) -> Option<QuantizedLpc> {
    debug_assert!((MIN_PRECISION..=MAX_PRECISION).contains(&precision));
    let cmax = a.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if !cmax.is_finite() || cmax < 1e-9 { return None; }
    let log2cmax = detmath::floor_log2(cmax);
    let shift = ((precision as i32 - 1) - log2cmax - 1).clamp(0, MAX_SHIFT as i32) as u32;
    let qmax = (1i64 << (precision - 1)) - 1;
    let qmin = -(1i64 << (precision - 1));
    let scale = (1i64 << shift) as f64;
    let mut error = 0.0f64;
    let mut coeffs = Vec::with_capacity(a.len());
    for &c in a {
        let v = c * scale + error;
        let qi = v.round().clamp(qmin as f64, qmax as f64) as i64;
        error = v - qi as f64;
        coeffs.push(qi);
    }
    Some(QuantizedLpc { coeffs, shift, precision })
}

/// [`analytic_bits`]'s second model: the order-2 fixed predictor's residual Rice-costed per
/// 256-sample partition (k from each partition's mean, `2|r|` standing in for the zigzag value).
/// Unlike the variance model it sees sparse residuals -- runs of exact prediction cost a 6-bit
/// escape per partition in `rice::encode`, far below the variance model's 1 bit/sample floor.
fn fixed2_partitioned_bits(samples: &[i64]) -> f64 {
    const PART: usize = 256;
    let n = samples.len();
    if n < 3 { return f64::INFINITY; }
    let mut bits = 2.0 * 25.0 + 8.0; // warmup (worst case, verbatim) + fields
    let res = (2..n).map(|i| samples[i] - 2 * samples[i - 1] + samples[i - 2]);
    let mut sum = 0u64;
    let mut len = 0usize;
    let flush = |sum: u64, len: usize, bits: &mut f64| {
        if len == 0 { return; }
        if sum == 0 { *bits += 11.0; return; }
        let mean = 2.0 * sum as f64 / len as f64;
        let k = if mean < 1.0 { 0 } else { detmath::floor_log2(mean).max(0) as u32 };
        *bits += 5.0 + len as f64 * (k as f64 + 1.0) + (2 * sum >> k) as f64;
    };
    for r in res {
        sum += r.unsigned_abs();
        len += 1;
        if len == PART { flush(sum, len, &mut bits); sum = 0; len = 0; }
    }
    flush(sum, len, &mut bits);
    bits
}

/// Per-coefficient side-info charge in [`analytic_bits`]. Calibrated on real recordings
/// (6/9/12/16 tried; 16 best, the spread across all tried
/// values was 0.09% of size).
const ANALYTIC_COEFF_BITS: f64 = 16.0;

/// Analytic bit-cost estimate for coding `samples` as one subframe (block-size search):
/// one Tukey(0.5) autocorrelation + Levinson-Durbin, no residuals, no Rice coding. Per order,
/// residual variance ~ (Levinson error / r0) x the raw mean square, costed as a Laplacian coded at
/// `0.5*log2(2 e^2 var)` bits/sample (floored at 1, Rice's minimum), plus coefficients and fields.
/// Only ever compared against itself over the same samples, so its constant offsets largely
/// cancel; the frames it picks are then really encoded.
pub fn analytic_bits(samples: &[i64], candidate_orders: &[usize]) -> f64 {
    let n = samples.len();
    if n == 0 { return 0.0; }
    // A Constant subframe (digital silence, a held value) costs one sample, not ~1 bit per sample.
    if samples.iter().all(|&x| x == samples[0]) { return 32.0; }
    let ms = samples.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n as f64;
    let per_sample = |var: f64| (0.5 * detmath::log2(2.0 * std::f64::consts::E * std::f64::consts::E * var.max(1e-3))).max(1.0);
    let mut best = (n as f64 * per_sample(ms) + 8.0).min(fixed2_partitioned_bits(samples));
    let max_order = MAX_ORDER.min(n.saturating_sub(1));
    if max_order == 0 || ms <= 0.0 { return if ms <= 0.0 { 16.0 } else { best }; }
    let w = windowed(samples, Window::Tukey50);
    let r = autocorr(&w, max_order);
    if r[0] <= 0.0 { return best; }
    let err = levinson_errors(&r, max_order);
    for &o in candidate_orders {
        if o == 0 || o > err.len() { continue; }
        let var = err[o - 1] / r[0] * ms;
        let bits = n as f64 * per_sample(var) + o as f64 * ANALYTIC_COEFF_BITS + 17.0;
        if bits < best { best = bits; }
    }
    best
}

/// How many of `candidate_orders` get a real quantized candidate returned, after ranking by the
/// Levinson-Durbin error estimate. >1 as a hedge against the estimate's own inaccuracy (it's a
/// per-block analytic proxy, not the real Rice-coded cost) -- see `candidates`.
const LPC_SHORTLIST: usize = 3;

/// Of the up to `LPC_SHORTLIST * WINDOWS.len()` candidates, only the best `LPC_TOP` by the analytic
/// estimate are costed on real residuals: 8 of 12 costs +0.007% size for ~8% encode time at `Max`
/// (6 costs +0.07%). `FAK_LPC_TOP` overrides (for testing).
const LPC_TOP: usize = 8;

/// For a block, try Levinson-Durbin once per window in `WINDOWS` (one autocorrelation pass each),
/// then use each recursion's own predicted-error sequence -- a free byproduct, no residuals
/// materialized -- to rank `candidate_orders` within that window and quantize only its top
/// `LPC_SHORTLIST`. This exists because computing real integer residuals for every candidate order
/// (the previous approach) was the dominant cost in LPC search; ranking analytically first cuts
/// each window's search from ~9 orders to `LPC_SHORTLIST`. Callers still cost-compare the combined
/// shortlist (up to `LPC_SHORTLIST * WINDOWS.len()` candidates) directly on real residuals -- this
/// only prunes which (window, order) pairs reach that stage, it doesn't skip real cost-comparison.
pub fn candidates(samples: &[i64], candidate_orders: &[usize], bits_eff: u32, history_len: usize) -> Vec<QuantizedLpc> {
    candidates_float(samples, candidate_orders, bits_eff, history_len).iter().filter_map(|a| quantize(a)).collect()
}

/// [`candidates`] before quantization: the float coefficients, so a caller can requantize the
/// winner at other precisions.
pub fn candidates_float(samples: &[i64], candidate_orders: &[usize], bits_eff: u32, history_len: usize) -> Vec<Vec<f64>> {
    candidates_float_in(samples, candidate_orders, bits_eff, history_len, active_windows()).floats
}

/// Where a candidate's coefficients came from: the window (index into [`CandidateSet::autocorrs`]) and
/// the Levinson prediction error at its order, in the units of that window's autocorrelation.
#[derive(Clone, Copy)]
pub struct CandModel { pub window: usize, pub err: f64 }

/// The LPC candidates of a block with what the encoder needs to reason about them without residuals.
pub struct CandidateSet {
    pub floats: Vec<Vec<f64>>,
    pub model: Vec<CandModel>,
    /// Windowed autocorrelation (lags `0..=max_order`) of each window that produced candidates.
    pub autocorrs: Vec<Vec<f64>>,
    /// Position of the candidate with the smallest analytic bit estimate, and that estimate (the
    /// Levinson error model: coefficients, warmup and `n * 0.5 * log2(err)`).
    pub best_pos: usize,
    pub best_est: f64,
}

/// [`candidates_float`] with its analysis attached.
pub fn candidates_full(samples: &[i64], candidate_orders: &[usize], bits_eff: u32, history_len: usize) -> CandidateSet {
    candidates_float_in(samples, candidate_orders, bits_eff, history_len, active_windows())
}

/// [`quantize_at`] into a stack array (no allocation); returns the shift, or `None` as `quantize_at` does.
fn quantize_into(a: &[f64], precision: u32, out: &mut [i64; MAX_ORDER]) -> Option<u32> {
    let cmax = a.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if !cmax.is_finite() || cmax < 1e-9 { return None; }
    let log2cmax = detmath::floor_log2(cmax);
    let shift = ((precision as i32 - 1) - log2cmax - 1).clamp(0, MAX_SHIFT as i32) as u32;
    let qmax = (1i64 << (precision - 1)) - 1;
    let qmin = -(1i64 << (precision - 1));
    let scale = (1i64 << shift) as f64;
    let mut error = 0.0f64;
    for (o, &c) in out.iter_mut().zip(a) {
        let v = c * scale + error;
        let qi = v.round().clamp(qmin as f64, qmax as f64) as i64;
        error = v - qi as f64;
        *o = qi;
    }
    Some(shift)
}

/// Approximately [`coeff_bits`]: each group's Rice parameter searched by walking from the one its mean
/// suggests instead of scanning all 16 (the cost is convex enough in `k` that this finds the minimum;
/// only ever used to compare precisions, never to write).
fn coeff_bits_est(c: &[i64]) -> u64 {
    fn group(g: &[i64]) -> u64 {
        if g.is_empty() { return 0; }
        let cost = |k: u32| g.iter().map(|&v| (crate::rice::zigzag(v) >> k) + 1 + k as u64).sum::<u64>();
        let mean = g.iter().map(|&v| crate::rice::zigzag(v)).sum::<u64>() / g.len() as u64;
        let mut k = (64 - mean.leading_zeros()).saturating_sub(1).min(15);
        let mut best = cost(k);
        loop {
            let (lo, hi) = (if k > 0 { cost(k - 1) } else { u64::MAX }, if k < 15 { cost(k + 1) } else { u64::MAX });
            if lo < best && lo <= hi { k -= 1; best = lo; } else if hi < best { k += 1; best = hi; } else { break; }
        }
        4 + best
    }
    let (a, b) = coeff_groups(c);
    group(a) + group(b)
}

/// The coefficient precision the excess-power model ([`precision_excess`]) prefers for candidate `a`
/// (solved from autocorrelation `r` with Levinson error `err`, `n_coded` coded samples), relative to
/// the reference quantization `q_ref` whose real cost the caller knows: the precision and the model's
/// predicted change in bits against `q_ref`. The model's cost is not unimodal in the precision (the
/// shift changes with it, and the chosen precisions are bimodal), so every second precision is
/// priced and then the two neighbours of the best.
pub fn best_precision(a: &[f64], r: &[f64], err: f64, q_ref: &QuantizedLpc, n_coded: f64) -> Option<(u32, f64)> {
    let p = a.len();
    if p > MAX_ORDER || p == 0 { return None; }
    // Model cost of quantized coefficients, up to the constant shared by all: coefficient bits + the
    // bits the excess power `d' R d` adds. With `c_l = sum_j d_j d_{j+l}` that quadratic form is
    // `r_0 c_0 + 2 sum_l r_l c_l`.
    let model = |coeffs: &[i64], shift: u32| -> f64 {
        let inv = 1.0 / (1u64 << shift) as f64;
        let mut d = [0.0f64; MAX_ORDER];
        for j in 0..p { d[j] = coeffs[j] as f64 * inv - a[j]; }
        let mut x = 0.0;
        for l in 0..p {
            let mut c = 0.0;
            for j in 0..p - l { c += d[j] * d[j + l]; }
            x += if l == 0 { r[0] * c } else { 2.0 * r[l] * c };
        }
        coeff_bits_est(&coeffs[..p]) as f64 + 0.5 * n_coded * detmath::log2(err + x)
    };
    let mut buf = [0i64; MAX_ORDER];
    let cost = |prec: u32, buf: &mut [i64; MAX_ORDER]| -> Option<f64> { let shift = quantize_into(a, prec, buf)?; Some(model(&buf[..p], shift)) };
    let reference = model(&q_ref.coeffs, q_ref.shift);
    let mut best = (f64::INFINITY, 0u32);
    let consider = |prec: u32, buf: &mut [i64; MAX_ORDER], best: &mut (f64, u32)| {
        if let Some(c) = cost(prec, buf) { if c < best.0 { *best = (c, prec); } }
    };
    for prec in (MIN_PRECISION + 1..=MAX_PRECISION).step_by(2) { consider(prec, &mut buf, &mut best); }
    if best.1 == 0 { return None; }
    for prec in [best.1 - 1, best.1 + 1] {
        if (MIN_PRECISION..=MAX_PRECISION).contains(&prec) { consider(prec, &mut buf, &mut best); }
    }
    if best.1 == q_ref.precision { return None; }
    Some((best.1, best.0 - reference))
}

/// Extra residual power (in the units of `r`, the autocorrelation the coefficients `a` were solved
/// from) from using the quantized coefficients `q` instead of `a`: `d' R d` for `d = q - a` and `R` the
/// Toeplitz matrix of `r`. For coefficients that minimize the windowed error this is exact to second
/// order, so it prices a precision without computing a residual.
pub fn precision_excess(q: &QuantizedLpc, a: &[f64], r: &[f64]) -> f64 {
    let inv = 1.0 / (1u64 << q.shift) as f64;
    let d: Vec<f64> = q.coeffs.iter().zip(a).map(|(&c, &x)| c as f64 * inv - x).collect();
    let mut s = 0.0;
    for j in 0..d.len() {
        let mut row = 0.0;
        for k in 0..d.len() { row += d[k] * r[j.abs_diff(k)]; }
        s += d[j] * row;
    }
    s
}


fn candidates_float_in(samples: &[i64], candidate_orders: &[usize], bits_eff: u32, history_len: usize, windows: &[Window]) -> CandidateSet {
    let max_order = MAX_ORDER.min(samples.len().saturating_sub(1));
    if max_order == 0 { return CandidateSet { floats: Vec::new(), model: Vec::new(), autocorrs: Vec::new(), best_pos: 0, best_est: f64::INFINITY }; }
    let n = samples.len() as f64;
    let mut out: Vec<(f64, Vec<f64>, CandModel)> = Vec::with_capacity(LPC_SHORTLIST * WINDOWS.len());
    let mut autocorrs: Vec<Vec<f64>> = Vec::with_capacity(windows.len());
    for &kind in windows {
        let g = crate::prof::span(crate::prof::Phase::Autocorr);
        let r = autocorr(&windowed(samples, kind), max_order);
        drop(g);
        if r[0] <= 0.0 { continue; }
        let g = crate::prof::span(crate::prof::Phase::Levinson);
        let (by_order, err_by_order) = levinson_all_orders(&r, max_order);
        drop(g);
        let _g = crate::prof::span(crate::prof::Phase::CandidateRanking);
        let window = autocorrs.len();
        autocorrs.push(r);
        let mut scored: Vec<(usize, f64)> = candidate_orders.iter()
            .filter(|&&o| o >= 1 && o <= by_order.len())
            .map(|&o| {
                let err = err_by_order[o - 1].max(1e-9);
                let (warmup_bits, coded) = if history_len >= o { (0.0, n) } else { (bits_eff as f64, n - o as f64) };
                let side_info = o as f64 * (warmup_bits + PRECISION as f64) + 5.0;
                let est = side_info + coded.max(0.0) * 0.5 * detmath::log2(err);
                (o, est)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(LPC_SHORTLIST);
        out.extend(scored.into_iter().map(|(o, est)| (est, by_order[o - 1].clone(), CandModel { window, err: err_by_order[o - 1].max(1e-9) })));
    }
    let top = std::env::var("FAK_LPC_TOP").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(LPC_TOP);
    if out.len() > top {
        // Keep the `top` best by the analytic estimate, in their original order.
        let mut ranked: Vec<f64> = out.iter().map(|c| c.0).collect();
        ranked.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cut = ranked[top - 1];
        let mut kept = 0;
        out.retain(|c| { let k = c.0 <= cut && kept < top; if k { kept += 1; } k });
    }
    // Position of the candidate with the best analytic estimate, among those kept.
    let (best_pos, best_est) = out.iter().enumerate().min_by(|a, b| a.1.0.total_cmp(&b.1.0)).map_or((0, f64::INFINITY), |(i, c)| (i, c.0));
    let model = out.iter().map(|c| c.2).collect();
    CandidateSet { floats: out.into_iter().map(|c| c.1).collect(), model, autocorrs, best_pos, best_est }
}

/// Residuals for samples[order..] (warmup = samples[..order]).
///
/// `i64` accumulation, not `i128` like `reconstruct`'s own inline dot product: this is encode-only
/// (never on the decode path), where `samples` always holds real input -- bounded by this codec's
/// max 24-bit depth (`format.rs`/`wav.rs`), so at most `2^24` in magnitude even for the widened
/// stereo side channel (`stereo::side`, `l - r`). With `|coeff| <= 2^14` (at most `MAX_PRECISION`-bit
/// quantization, `quantize`) and `order <= MAX_ORDER = 32`, the worst-case sum magnitude is
/// `<= 32 * 2^14 * 2^24 = 2^43` -- about `2^20`x inside `i64::MAX`, nowhere near overflow (bounds proven, not assumed). `reconstruct` keeps `i128` deliberately: there, the history
/// holds *reconstructed* samples fed back from a stream that could be hostile, bounded only by the
/// much looser defensive `SANE_SAMPLE_BOUND` (`2^48`, `predictors.rs`) until the next post-hoc check
/// rejects it -- `32 * 2^14 * 2^48 = 2^67` genuinely would overflow `i64`.
///
/// The candidate search calls this for every shortlisted (window, order) pair of every channel of
/// every block: ~40-67% of block-mode encode instructions. Implemented in `simd::lpc_residuals` (AVX-512F/AVX2 kernels,
/// runtime-dispatched, against a scalar reference); every path gives identical output.
pub fn residuals(q: &QuantizedLpc, samples: &[i64]) -> Vec<i64> {
    crate::simd::lpc_residuals(&q.coeffs, q.shift, samples)
}

/// [`residuals`] for costing candidates only (`simd::lpc_residuals_estimate`): never coded.
pub fn residuals_estimate(q: &QuantizedLpc, samples: &[i64]) -> Vec<i64> {
    crate::simd::lpc_residuals_estimate(&q.coeffs, q.shift, samples)
}

/// Reconstruct the full sample sequence. Bounds each reconstructed sample the same way
/// `predictors::reconstruct` does, for the same reason (a hostile stream must not be able to
/// inject a residual that compounds through the feedback loop into an overflow or a hang).
pub fn reconstruct(q: &QuantizedLpc, warmup: &[i64], res: &[i64]) -> Result<Vec<i64>, &'static str> {
    let mut out = Vec::with_capacity(warmup.len() + res.len());
    out.extend_from_slice(warmup);
    reconstruct_into(q, warmup, res, &mut out)?;
    Ok(out)
}

/// [`reconstruct`] without the warmup: appends only the `res.len()` new samples to `out`. Used
/// when the warmup is history from the preceding frame (format v9) and so already decoded.
pub fn reconstruct_into(q: &QuantizedLpc, warmup: &[i64], res: &[i64], out: &mut Vec<i64>) -> Result<(), &'static str> {
    let order = q.coeffs.len();
    debug_assert_eq!(warmup.len(), order);
    // Windowed form (the same restructuring `residuals` got in): `buf` holds the
    // warmup followed by every sample reconstructed so far, and each prediction reads its `order`
    // predecessors straight out of it -- no per-sample shift of a history buffer, which cost as
    // many moves as the dot product has multiplies. `rev[k] = coeffs[order-1-k]` lines the
    // coefficients up with the window oldest-first. Same i128 arithmetic as before, so the same
    // bound argument (doc comment above) holds and the output is identical.
    //
    // Fast path first (`simd::lpc_reconstruct_i32`: exact i32 x i32 -> i64, one unrolled loop per
    // order): every legitimate stream stays inside it. The i128 loop below finishes whatever it hands back --
    // only reachable by a hostile stream, where it applies the same bound as before.
    let mut done = 0;
    let mut buf: Vec<i64> = Vec::with_capacity(order + res.len());
    let fits = |v: &i64| *v == *v as i32 as i64;
    if q.coeffs.iter().all(fits) && warmup.iter().all(fits) && q.shift < 64 {
        let rev32: Vec<i32> = q.coeffs.iter().rev().map(|&c| c as i32).collect();
        let mut buf32: Vec<i32> = Vec::with_capacity(order + res.len());
        buf32.extend(warmup.iter().map(|&w| w as i32));
        done = crate::simd::lpc_reconstruct_i32(&rev32, q.shift, res, &mut buf32);
        if done == res.len() {
            out.extend(buf32[order..].iter().map(|&x| x as i64));
            return Ok(());
        }
        buf.extend(buf32.iter().map(|&x| x as i64));
    } else {
        buf.extend_from_slice(warmup);
    }
    let rev: Vec<i64> = q.coeffs.iter().rev().copied().collect();
    for (i, &e) in res.iter().enumerate().skip(done) {
        let win = &buf[i..i + order];
        let mut acc: i128 = 0;
        for (&c, &x) in rev.iter().zip(win) { acc += c as i128 * x as i128; }
        let x128 = e as i128 + (acc >> q.shift);
        if x128.abs() > SANE_SAMPLE_BOUND { return Err("reconstructed sample out of sane range (corrupted stream?)"); }
        buf.push(x128 as i64);
    }
    out.extend_from_slice(&buf[order..]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A noisy AR(3) signal, deterministic, in the 24-bit range.
    fn ar_signal(n: usize, seed: u64) -> Vec<i64> {
        let mut st = seed;
        let mut next = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
        let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
        (0..n).map(|_| { let x = 2.6 * a - 2.3 * b + 0.68 * c + 300.0 * next(); c = b; b = a; a = x; (x * 40.0) as i64 }).collect()
    }

    #[test]
    fn precision_model_chooses_no_worse_than_the_reference_on_real_residuals() {
        // `best_precision` prices precisions without residuals; its pick, costed on the real residual, must
        // not be worse than the reference precision by more than the estimate's own noise (1% of the bits).
        for seed in [3u64, 11, 29] {
            let x = ar_signal(4096, seed);
            let set = candidates_full(&x, &[4, 8, 16, 24, 32], 24, 0);
            assert!(!set.floats.is_empty());
            let real = |q: &QuantizedLpc| coeff_bits(&q.coeffs) + crate::rice::estimate_bits(&residuals(q, &x));
            for (i, a) in set.floats.iter().enumerate() {
                let q_ref = quantize(a).expect("quantizes");
                let m = set.model[i];
                let picked = best_precision(a, &set.autocorrs[m.window], m.err, &q_ref, x.len() as f64);
                if let Some((prec, _)) = picked {
                    assert!((MIN_PRECISION..=MAX_PRECISION).contains(&prec));
                    let q = quantize_at(a, prec).expect("quantizes at the pick");
                    let (bits_ref, bits) = (real(&q_ref) as f64, real(&q) as f64);
                    assert!(bits <= bits_ref * 1.01, "seed {seed} candidate {i} order {}: precision {prec} costs {bits} vs {bits_ref} at the reference", a.len());
                }
            }
        }
    }

    #[test]
    fn excess_power_is_small_at_high_precision_and_shrinks_with_it() {
        let x = ar_signal(4096, 5);
        let set = candidates_full(&x, &[8, 16], 24, 0);
        let (a, m) = (&set.floats[0], set.model[0]);
        let r = &set.autocorrs[m.window];
        let excess = |p: u32| precision_excess(&quantize_at(a, p).unwrap(), a, r);
        assert!(excess(16) >= 0.0);
        assert!(excess(6) > excess(10) && excess(10) > excess(14), "{} {} {}", excess(6), excess(10), excess(14));
        // Relative to the prediction error itself the excess at 16 bits is negligible (about -55 dB here).
        assert!(excess(16) < 1e-4 * m.err.max(1.0), "{} vs {}", excess(16), m.err);
    }

    #[test]
    fn coeff_bits_estimate_is_the_exact_minimum_or_just_above_it() {
        let x = ar_signal(4096, 17);
        for a in candidates_full(&x, &[2, 6, 12, 24, 32], 24, 0).floats {
            for p in [4u32, 8, 12, 16] {
                let q = quantize_at(&a, p).unwrap();
                let (est, exact) = (coeff_bits_est(&q.coeffs), coeff_bits(&q.coeffs));
                assert!(est >= exact, "an estimate cannot beat the true minimum: {est} < {exact}");
                assert!(est <= exact + exact / 25 + 2, "order {} precision {p}: {est} vs {exact}", a.len());
            }
        }
    }

    /// The history-buffer form of `residuals`, kept verbatim as the reference the
    /// windowed form must match exactly.
    fn residuals_reference(q: &QuantizedLpc, samples: &[i64]) -> Vec<i64> {
        let order = q.coeffs.len();
        let n = samples.len();
        if n <= order { return Vec::new(); }
        let mut out = Vec::with_capacity(n - order);
        let mut h = vec![0i64; order];
        for k in 0..order { h[k] = samples[order - 1 - k]; }
        for i in order..n {
            let acc: i64 = q.coeffs.iter().zip(&h).map(|(&c, &x)| c * x).sum();
            out.push(samples[i] - (acc >> q.shift));
            for k in (1..order).rev() { h[k] = h[k - 1]; }
            h[0] = samples[i];
        }
        out
    }

    #[test]
    fn window_cache_matches_direct_computation() {
        // The cache indexes its tables by `Window` discriminant; that only works if `ALL_WINDOWS`
        // lists the variants in discriminant order.
        for (i, &k) in ALL_WINDOWS.iter().enumerate() { assert_eq!(k as usize, i); }
        let samples: Vec<i64> = (0..257i64).map(|i| (i * 7919) % 2001 - 1000).collect();
        for len in [2usize, 3, 256, 257, 256, 2] { // revisits lengths: cache refill paths
            for &k in ALL_WINDOWS {
                let direct: Vec<f64> = (0..len).map(|i| samples[i] as f64 * window_fn(k, len, i)).collect();
                let cached = windowed(&samples[..len], k);
                assert_eq!(cached.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                           direct.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "len={len}");
            }
        }
    }

    #[test]
    fn windowed_residuals_match_history_buffer_reference() {
        // xorshift PRNG: deterministic, no dependency.
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for trial in 0..400 {
            let n = 1 + (next() % 300) as usize;
            let bits = [8u32, 16, 24, 25][trial % 4]; // 25: the widened stereo side channel
            let lim = 1i64 << (bits - 1);
            let samples: Vec<i64> = (0..n).map(|_| (next() as i64).rem_euclid(2 * lim) - lim).collect();
            for order in 1..=MAX_ORDER {
                // Extreme coefficients included: every one at the +/- MAX_PRECISION-bit limit.
                let coeffs: Vec<i64> = (0..order).map(|j| match (trial + j) % 5 {
                    0 => (1 << (MAX_PRECISION - 1)) - 1,
                    1 => -(1 << (MAX_PRECISION - 1)),
                    _ => (next() as i64).rem_euclid(1 << MAX_PRECISION) - (1 << (MAX_PRECISION - 1)),
                }).collect();
                let q = QuantizedLpc { coeffs, shift: (next() % (MAX_SHIFT as u64 + 1)) as u32, precision: MAX_PRECISION };
                assert_eq!(residuals(&q, &samples), residuals_reference(&q, &samples), "n={n} order={order}");
            }
        }
    }

    fn roundtrip(samples: &[i64], candidate_orders: &[usize]) {
        for q in candidates(samples, candidate_orders, 16, 0) {
            let order = q.coeffs.len();
            let res = residuals(&q, samples);
            let back = reconstruct(&q, &samples[..order], &res).unwrap();
            assert_eq!(back, samples, "order {order} mismatch");
        }
    }

    #[test]
    fn roundtrip_on_tonal_and_noisy_signals() {
        let orders = [1, 2, 4, 8, 12, 16, 24, 32];
        let tonal: Vec<i64> = (0..4000).map(|i| (2000.0 * (i as f64 * 0.05).sin()) as i64).collect();
        roundtrip(&tonal, &orders);
        let noisy: Vec<i64> = (0..4000).map(|i: i64| ((i * 92821) % 40001) - 20000).collect();
        roundtrip(&noisy, &orders);
        let extreme_24bit: Vec<i64> = (0..2000).map(|i| if i % 2 == 0 { 8_388_607 } else { -8_388_608 }).collect();
        roundtrip(&extreme_24bit, &orders);
    }

    #[test]
    fn degenerate_blocks_dont_panic() {
        assert!(candidates(&[], &[1, 4, 8], 16, 0).is_empty());
        assert!(candidates(&[5], &[1, 4, 8], 16, 0).is_empty());
        let _ = candidates(&vec![0i64; 100], &[1, 4, 8], 16, 0); // silence: must not panic, whatever it returns
        let _ = candidates(&vec![7i64; 100], &[1, 4, 8], 16, 32); // constant nonzero, zero variance after order 1
    }

    /// The pre-v9 shifting-history-buffer form of `reconstruct`, kept verbatim as the reference
    /// the windowed `reconstruct_into` must match exactly, errors included.
    fn reconstruct_reference(q: &QuantizedLpc, warmup: &[i64], res: &[i64]) -> Result<Vec<i64>, &'static str> {
        let order = q.coeffs.len();
        let mut out = Vec::with_capacity(order + res.len());
        out.extend_from_slice(warmup);
        let mut h = vec![0i64; order];
        for k in 0..order { h[k] = warmup[order - 1 - k]; }
        for &e in res {
            let mut acc: i128 = 0;
            for (j, &c) in q.coeffs.iter().enumerate() { acc += c as i128 * h[j] as i128; }
            let x128 = e as i128 + (acc >> q.shift);
            if x128.abs() > SANE_SAMPLE_BOUND { return Err("reconstructed sample out of sane range (corrupted stream?)"); }
            let x = x128 as i64;
            out.push(x);
            for k in (1..order).rev() { h[k] = h[k - 1]; }
            h[0] = x;
        }
        Ok(out)
    }

    #[test]
    fn windowed_reconstruct_matches_history_buffer_reference() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for trial in 0..600 {
            let order = 1 + trial % MAX_ORDER;
            let prec = MIN_PRECISION + (trial as u32 % (MAX_PRECISION - MIN_PRECISION + 1));
            let lim = 1i64 << (prec - 1);
            let coeffs: Vec<i64> = (0..order).map(|_| (next() as i64).rem_euclid(2 * lim) - lim).collect();
            let q = QuantizedLpc { coeffs, shift: (next() % (MAX_SHIFT as u64 + 1)) as u32, precision: prec };
            // Warmup magnitudes up to history's hostile worst case (2^51); residuals up to a 40-bit
            // escape. Most trials hit the bound check part-way, some run to the end.
            let wbits = [8u32, 16, 25, 40, 52][trial % 5];
            let warmup: Vec<i64> = (0..order).map(|_| (next() as i64) >> (64 - wbits)).collect();
            let rbits = [1u32, 8, 20, 40][(trial / 5) % 4];
            let res: Vec<i64> = (0..(next() % 200) as usize).map(|_| (next() as i64) >> (64 - rbits)).collect();
            let mut got = warmup.clone();
            let r = reconstruct_into(&q, &warmup, &res, &mut got).map(|_| got);
            assert_eq!(r, reconstruct_reference(&q, &warmup, &res), "trial {trial}");
            // The i32 fast path itself, with its i128 hand-off, at every order (the threshold only
            // decides whether it is used): same values up to where it stops, and it stops exactly
            // at the first value outside i32.
            if q.coeffs.iter().chain(&warmup).all(|&v| v == v as i32 as i64) {
                let rev32: Vec<i32> = q.coeffs.iter().rev().map(|&c| c as i32).collect();
                let mut buf32: Vec<i32> = warmup.iter().map(|&w| w as i32).collect();
                let done = crate::simd::lpc_reconstruct_i32(&rev32, q.shift, &res, &mut buf32);
                let reference = reconstruct_reference(&q, &warmup, &res);
                let full = match &reference { Ok(v) => v.clone(), Err(_) => Vec::new() };
                for (k, &x) in buf32[order..].iter().enumerate() { if k < full.len().saturating_sub(order) { assert_eq!(x as i64, full[order + k], "trial {trial}"); } }
                if done < res.len() && full.len() == order + res.len() { assert!(full[order + done] != full[order + done] as i32 as i64, "trial {trial}"); }
            }
        }
    }

    #[test]
    fn reconstruct_rejects_runaway_hostile_residuals() {
        let q = QuantizedLpc { coeffs: vec![8191, -8191, 8191, -8191], shift: 0, precision: PRECISION };
        let huge = 1i64 << 60;
        assert!(reconstruct(&q, &[0, 0, 0, 0], &[huge, huge, huge]).is_err());
    }
}
