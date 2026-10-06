//! Lossless value map: a per-chunk, per-channel monotone map from a coarser integer
//! signal `k` to the stored samples,
//!
//! ```text
//! x = clamp(floor((k * P + C) / 2^32), lo, hi)      P > 2^32, 0 <= C < P
//! ```
//!
//! where `[lo, hi]` is the container's sample range. It covers PCM whose values lie on a structure
//! finer than FLAC-style wasted bits can express: a non-power-of-two integer step (`x = 3k`), an
//! offset lattice (`x = 256k + 128`), and a fixed digital gain `G = P / 2^32 > 1` applied to
//! lower-resolution material without dither (`x = round(k * G)`, e.g. a 16-bit master exported to a
//! 24-bit container at -1 dB). The clamp makes clipped samples representable. Since format 11
//! a map may carry sparse corrections, `x = clamp(raw(k) + e)` at listed positions, so
//! material that is only *almost* on a lattice (up to 10% of samples off it) maps too. The decoder
//! only evaluates the formula; everything else here is encoder-side detection. Nothing is
//! discarded: every sample the map does not reproduce exactly carries a correction.

pub const SHIFT: u32 = 32;
const ONE: i128 = 1 << SHIFT;
/// Clamp on a decoded `k` before mapping (see `ValueMap::apply_ref`).
const K_LIMIT: i64 = 1 << 25;

/// Distinct values needed before a gain is estimated (fewer are the palette's territory, and a
/// gain fitted from a handful of points is not worth trusting).
const MIN_DISTINCT: usize = 64;
/// Distinct values in the central window the first gain estimate comes from.
const WINDOW: usize = 257;
/// A first estimate below `1 + 1/MIN_GAIN_RECIP` is treated as "no structure" (dense values);
/// saving under ~0.02 bit/sample is not worth a map.
const MIN_GAIN_RECIP: i64 = 64;
/// Width of the pre-filter's value window (`plausible`), in sample values.
const PREFILTER_WINDOW: i64 = 4096;
/// Half-width of the P search around the least-squares estimate, in units of `2^-32`.
const P_SEARCH: i128 = 1 << 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueMap { pub p: u64, pub c: u64 }

impl ValueMap {
    /// Decoder side, reference form (exact i128). `k` is clamped first so the product cannot
    /// overflow on hostile input (any `|k| >= 2^25` lands outside every container range anyway,
    /// since `P > 2^32` and containers are at most 24 bits).
    pub fn apply_ref(&self, k: i64, lo: i64, hi: i64) -> i64 {
        let k = k.clamp(-K_LIMIT, K_LIMIT) as i128;
        let v = (k * self.p as i128 + self.c as i128) >> SHIFT;
        v.clamp(lo as i128, hi as i128) as i64
    }

    /// Decoder side, the same value in i64 before the clamp: with P = q*2^32 + f and
    /// C = cq*2^32 + cf, floor((k*P + C) / 2^32) = q*k + cq + floor((k*f + cf) / 2^32) exactly
    /// (the dropped terms are multiples of 2^32). With |k| <= 2^25 every term fits: |k*f| < 2^57,
    /// |q*k| < 2^57, and the result is within 2^58.
    #[inline]
    pub fn raw(&self, k: i64) -> i64 {
        let k = k.clamp(-K_LIMIT, K_LIMIT);
        let (q, f) = ((self.p >> SHIFT) as i64, (self.p & 0xFFFF_FFFF) as i64);
        let (cq, cf) = ((self.c >> SHIFT) as i64, (self.c & 0xFFFF_FFFF) as i64);
        q * k + cq + ((k * f + cf) >> SHIFT)
    }

    /// Decoder side: `raw(k)` clamped to the container range.
    #[inline]
    pub fn apply(&self, k: i64, lo: i64, hi: i64) -> i64 { self.raw(k).clamp(lo, hi) }

    /// Encoder side: the `k` that `apply` maps back to `x`, or `None` if there is none.
    pub fn forward(&self, x: i64, lo: i64, hi: i64) -> Option<i64> {
        let (p, c) = (self.p as i128, self.c as i128);
        let num = x as i128 * ONE - c;
        // Smallest k with k*P + C >= x*2^32; for x == lo the largest k with k*P + C < (lo+1)*2^32
        // (both then clamp to x when x is a clipped extreme).
        let k = if x == lo { ((lo as i128 + 1) * ONE - c - 1).div_euclid(p) } else { -((-num).div_euclid(p)) };
        let k = i64::try_from(k).ok()?;
        (self.apply(k, lo, hi) == x).then_some(k)
    }

    /// Encoder side: the k whose unclamped value is nearest `x` (ties to the smaller k).
    pub fn nearest(&self, x: i64) -> i64 {
        let (p, c) = (self.p as i128, self.c as i128);
        // Smallest k with k*P + C >= x*2^32, i.e. raw(k) >= x; its predecessor is the other side.
        let k = (-((-(x as i128 * ONE - c)).div_euclid(p))).clamp(-K_LIMIT as i128 + 1, K_LIMIT as i128) as i64;
        if (self.raw(k) - x).abs() < (x - self.raw(k - 1)).abs() { k } else { k - 1 }
    }

    pub fn is_valid(&self) -> bool { (self.p as i128) > ONE && self.c < self.p }
}

/// A channel's map for one chunk plus its sparse corrections (flag 2 of the section): at each
/// listed sample position (ascending, distinct) the output is `clamp(raw(k) + e)` instead of
/// `clamp(raw(k))`. Lets a map cover material that is *almost* on a lattice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMap { pub map: ValueMap, pub corr: Vec<(u32, i64)> }

impl ChannelMap {
    /// Whether this map is certain to shrink a chunk of `n` samples, so the encoder may skip
    /// encoding it both ways. In quarter-bits per sample: the map saves at least
    /// floor(4 log2 G); `c` corrections cost at most about (c/n)(log2(n/c) + 7) (a Rice-coded
    /// position gap near log2(n/c) + 2 bits, a value a few bits), bounded above here with
    /// integer log2. Pays clearly when the saving beats that by a quarter-bit or more.
    pub fn clearly_pays(&self, n: usize) -> bool {
        let saved = QUARTER_BIT_P.iter().rposition(|&t| self.map.p >= t).unwrap_or(0) as u64;
        let c = self.corr.len() as u64;
        let n = n.max(1) as u64;
        let cost = if c == 0 { 0 } else { (4 * c * ((n / c).ilog2() as u64 + 8)).div_ceil(n) };
        saved >= cost + 1
    }

    /// Decoder side: replaces a decoded channel's `k` values by output samples.
    pub fn apply_all(&self, chan: &mut [i64], lo: i64, hi: i64) {
        // Corrections first, while chan still holds k (read_section checked every position).
        let fixed: Vec<i64> = self.corr.iter().map(|&(i, e)| (self.map.raw(chan[i as usize]) + e).clamp(lo, hi)).collect();
        for s in chan.iter_mut() { *s = self.map.apply(*s, lo, hi); }
        for (&(i, _), v) in self.corr.iter().zip(fixed) { chan[i as usize] = v; }
    }
}

/// `QUARTER_BIT_P[j]` = ceil(2^(32 + j/4)): the smallest `P` of a gain of at least 2^(j/4), exact
/// integers (smallest t with t^4 >= 2^(128 + j)), so the encoder's decisions from it are the same
/// on every platform (no libm `log2`).
const QUARTER_BIT_P: [u64; 33] = [
    4294967296, 5107605668, 6074001000, 7223245206,
    8589934592, 10215211335, 12148002000, 14446490412,
    17179869184, 20430422669, 24296004000, 28892980823,
    34359738368, 40860845337, 48592008000, 57785961646,
    68719476736, 81721690674, 97184016000, 115571923291,
    137438953472, 163443381348, 194368031999, 231143846582,
    274877906944, 326886762695, 388736063997, 462287693164,
    549755813888, 653773525390, 777472127994, 924575386327,
    1099511627776,
];
/// Largest |correction| a decoder accepts (encoder corrections are a few units: the distance from
/// a sample to its nearest lattice point).
const MAX_CORRECTION: i64 = 1 << 26;
/// Most outlying samples (fraction of the chunk) a near-map may leave to corrections; beyond this
/// the corrections cost more than the map saves.
const MAX_OUTLIER_FRAC: f64 = 0.10;

/// Signed bit width needed to hold `v`.
fn signed_bits(v: i64) -> u32 { 65 - if v < 0 { (!v).leading_zeros() } else { v.leading_zeros() } }

/// The used values recorded in a pre-filter bitset, ascending.
fn window_values(used: &[u64], start: i64) -> Vec<i64> {
    let mut out = Vec::new();
    for (i, &w) in used.iter().enumerate() {
        let mut w = w;
        while w != 0 { out.push(start + i as i64 * 64 + w.trailing_zeros() as i64); w &= w - 1; }
    }
    out
}

/// O(n) pre-filter, no sort: the used values inside a `PREFILTER_WINDOW`-wide window around the
/// densest part of the distribution must show a map's footprint. A map `floor(k*G + c)` uses a
/// Beatty-like set: every run of 64 consecutive values holds floor(64/G) or ceil(64/G) of them.
///   * Ordinary audio fills the window's densest 64-value words completely: rejected at once.
///   * Stride 1-2 (G < 3): the densest words must be *balanced* (popcounts within 1 of each
///     other) and at least a third full. Undersampled natural content (e.g. 24-bit) is randomly
///     occupied and spreads by several counts; unused k only lower a word's count, so the densest
///     words of a real map stay balanced.
///   * Stride >= 3: gaps between used values must be about a whole number of strides, up to an
///     eighth of them (stray values of a near-map).
fn plausible(x: &[i64], lo: i64, hi: i64) -> bool {
    if x.len() < MIN_DISTINCT { return false; }
    // Centre on the densest part of the distribution (a coarse 256-bin histogram), where unused k
    // are rarest -- not the mean, which a sine-like signal visits least.
    let (mn, mx) = x.iter().fold((i64::MAX, i64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let shift = (64 - ((mx - mn) as u64).leading_zeros()).saturating_sub(8);
    let mut hist = [0u32; 257];
    for &v in x { hist[((v - mn) >> shift) as usize] += 1; }
    let peak = (0..hist.len()).max_by_key(|&i| (hist[i], std::cmp::Reverse(i))).expect("non-empty");
    let centre = mn + ((peak as i64) << shift) + ((1i64 << shift) >> 1);
    let start = (centre - PREFILTER_WINDOW / 2).clamp(lo, (hi - PREFILTER_WINDOW + 1).max(lo));
    let mut used = [0u64; PREFILTER_WINDOW as usize / 64];
    for &v in x {
        let d = v.wrapping_sub(start);
        if (0..PREFILTER_WINDOW).contains(&d) { used[d as usize / 64] |= 1 << (d % 64); }
    }
    let mut counts: Vec<u32> = used.iter().map(|w| w.count_ones()).collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    if counts[0] == 64 || counts[0] == 0 { return false; }
    let vals = window_values(&used, start);
    let gaps: Vec<i64> = vals.windows(2).map(|p| p[1] - p[0]).collect();
    if gaps.is_empty() { return false; }
    // The commonest gap picks the branch (stray values of a near-map cannot move it, unlike the
    // densest words' counts).
    let mut sorted = gaps.clone();
    sorted.sort_unstable();
    let (mut mode, mut best, mut i) = (sorted[0], 0, 0);
    while i < sorted.len() {
        let j = sorted[i..].partition_point(|&d| d == sorted[i]) + i;
        if j - i > best { best = j - i; mode = sorted[i]; }
        i = j;
    }
    if mode < 3 {
        // Stride 1-2: balanced, at least a third full.
        const TOP: usize = 8;
        return counts[0] >= 22 && counts[TOP - 1] > 0 && counts[0] - counts[TOP - 1] <= 1 && seed_line(&vals).is_some();
    }
    // Stride >= 3, from the commonest gap and its neighbours (so a non-integer G averages right);
    // unused k give gaps of whole strides.
    let near: Vec<i64> = gaps.iter().copied().filter(|&d| (d - mode).abs() <= 1).collect();
    let stride = near.iter().sum::<i64>() as f64 / near.len() as f64;
    // A power-of-two stride with every value on one residue 0 is plain wasted bits.
    if (mode as u64).is_power_of_two() && gaps.iter().all(|&d| d % mode == 0) && vals[0] % mode == 0 { return false; }
    // Off-lattice samples (up to MAX_OUTLIER_FRAC, left to corrections) add stray values; allow
    // an eighth of the gaps to be off.
    let bad = gaps.iter().filter(|&&d| { let m = (d as f64 / stride).round(); m < 1.0 || (d as f64 - m * stride).abs() > 1.0 + m * 0.01 }).count();
    bad <= gaps.len() / 8 && seed_line(&vals).is_some()
}

/// Least-squares slope and intercept of (k, x) points.
fn fit(pts: &mut dyn Iterator<Item = (f64, f64)>) -> Option<(f64, f64)> {
    let (mut n, mut sk, mut sx, mut skk, mut skx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (k, x) in pts { n += 1.0; sk += k; sx += x; skk += k * k; skx += k * x; }
    let den = n * skk - sk * sk;
    if den <= 0.0 { return None; }
    let g = (n * skx - sk * sx) / den;
    Some((g, (sx - g * sk) / n))
}

/// Least-squares line `x ~ G*k + b` through the densest `WINDOW` consecutive values of `v`
/// (sorted, distinct), with k seeded by counting strides; `None` unless it looks like a map
/// (G > 1 + 1/MIN_GAIN_RECIP, every window value within 0.75 of the line). Returns (G, b, start,
/// len) of the window.
fn seed_line(v: &[i64]) -> Option<(f64, f64, usize, usize)> {
    if v.len() < 3 { return None; }
    // First estimate: mean gap over the densest WINDOW consecutive distinct values -- the stretch
    // most likely to have every intermediate k in use (for a gain applied to low-resolution audio).
    let w = WINDOW.min(v.len());
    let a = (0..=v.len() - w).min_by_key(|&j| v[j + w - 1] - v[j]).expect("v.len() >= w");
    let span = v[a + w - 1] - v[a];
    if span * MIN_GAIN_RECIP < (w as i64 - 1) * (MIN_GAIN_RECIP + 1) { return None; }
    let gain0 = span as f64 / (w - 1) as f64;
    let win = &v[a..a + w];
    let (gain, icpt) = if gain0 < 3.0 {
        // Stride 1-2: k by rank (exact when the window is dense).
        fit(&mut win.iter().enumerate().map(|(i, &x)| (i as f64, x as f64)))?
    } else {
        // Stride >= 3: unused k make some gaps whole multiples, and stray off-lattice values (a
        // near-map's) split a gap in two, so neither ranks nor per-gap step counts are safe.
        // Start from the commonest gap's cluster, then assign k by absolute rounding against the
        // current line (a stray then misplaces only itself) over a doubling prefix, refitting
        // each round.
        let mut gaps: Vec<i64> = win.windows(2).map(|p| p[1] - p[0]).collect();
        gaps.sort_unstable();
        let mut mode = (gaps[0], 0usize);
        let mut i = 0;
        while i < gaps.len() {
            let j = gaps[i..].partition_point(|&d| d == gaps[i]) + i;
            if j - i > mode.1 { mode = (gaps[i], j - i); }
            i = j;
        }
        let near: Vec<i64> = gaps.iter().copied().filter(|&d| (d - mode.0).abs() <= 1).collect();
        let (mut g, mut b) = (near.iter().sum::<i64>() as f64 / near.len() as f64, win[0] as f64);
        let mut len = 16.min(w);
        loop {
            // All points first (so the slope follows the lattice's slow drift against the
            // current estimate), then only those within 0.75 of that line (dropping strays).
            let pts: Vec<(f64, f64)> = win[..len].iter().map(|&x| (((x as f64 - b) / g).round(), x as f64)).collect();
            let (g1, b1) = fit(&mut pts.iter().copied())?;
            let inl: Vec<(f64, f64)> = pts.into_iter().filter(|&(k, x)| (x - (g1 * k + b1)).abs() <= 0.75).collect();
            if inl.len() < 3 { return None; }
            (g, b) = fit(&mut inl.into_iter())?;
            if len == w { break; }
            len = (2 * len).min(w);
        }
        (g, b)
    };
    // Fail fast: a map's values sit within 1/2 of the line (plus the fit's own small error).
    // (An eighth of the window may be off: stray values a near-map leaves to corrections.)
    if v[a..a + w].iter().filter(|&&s| { let k = ((s as f64 - icpt) / gain).round(); (s as f64 - (gain * k + icpt)).abs() > 0.75 }).count() > w / 8 { return None; }
    Some((gain, icpt, a, w))
}

/// Finds a map for `x` (one channel of one chunk, container `bits`), or `None` when there is no
/// such structure. Cheap on ordinary audio: one sort, then an O(n) density test that dense
/// content fails immediately.
pub fn detect(x: &[i64], bits: u32) -> Option<(ChannelMap, Vec<i64>)> {
    //  this detector's cost model (the `seen` bitset below, `K_LIMIT`, `P_SEARCH`) assumes a
    // container of at most 24 bits, the widest this format supported when it was written -- a
    // 32-bit chunk's value range can span up to 2^32, which would size `seen` in the hundreds of MB
    // for ordinary noisy content. Simplest safe fix: no value-map detection above 24 bits: fall back
    // to plain Fixed/LPC/Cross + Rice, which already has correct headroom at any sample width.
    if bits > 24 { return None; }
    let (lo, hi) = (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1);
    if !plausible(x, lo, hi) { return None; }
    // Distinct values, ascending: a bitset over [min, max] (at most 2^24 wide for the containers
    // this covers) is far cheaper than sorting the chunk. Clipped extremes are left to the
    // clamp; fit the rest.
    let (mn, mx) = x.iter().fold((i64::MAX, i64::MIN), |(a, b), &s| (a.min(s), b.max(s)));
    let mut seen = vec![0u64; ((mx - mn) as usize >> 6) + 1];
    for &s in x { let d = (s - mn) as usize; seen[d >> 6] |= 1 << (d & 63); }
    let v: Vec<i64> = window_values(&seen, mn).into_iter().filter(|&s| s != lo && s != hi).collect();
    if v.len() < MIN_DISTINCT { return None; }

    let (mut gain, mut icpt, a, w) = seed_line(&v)?;
    let (mut l, mut r) = (a, a + w);
    while l > 0 || r < v.len() {
        let grow = ((r - l) / 2).max(1);
        l = l.saturating_sub(grow);
        r = (r + grow).min(v.len());
        let (g, b) = (gain, icpt);
        (gain, icpt) = fit(&mut v[l..r].iter().map(|&x| (((x as f64 - b) / g).round(), x as f64)))?;
        if !(gain > 1.0 + 1.0 / MIN_GAIN_RECIP as f64) || !gain.is_finite() { return None; }
    }

    // A map's values sit within 1/2 of the line (plus the fit's own small error). A near-map may
    // leave up to MAX_OUTLIER_FRAC of the *samples* off it, to be carried as corrections; beyond
    // that it is not worth it, and the exact search below would only fail slowly.
    let res = |s: i64| { let k = ((s as f64 - icpt) / gain).round(); (s as f64 - (gain * k + icpt)).abs() };
    let budget = (x.len() as f64 * MAX_OUTLIER_FRAC) as usize;
    let (mut outliers, mut clearly_off) = (0usize, false);
    for &s in x {
        if s == lo || s == hi { continue; }
        let r = res(s);
        if r > 0.5 { outliers += 1; if outliers > budget { return None; } }
        // (The fit's own error can put exact values a little past 1/2.)
        clearly_off |= r > 0.6;
    }

    // Exact fit on the inlier values: integers P, C with floor((k*P + C) / 2^32) == v. For a fixed
    // P the feasible C form an interval [L(P), U(P)]; U - L is concave in P, so a ternary search
    // around the least-squares P finds a feasible one whenever one exists nearby. Values within
    // rounding of the 1/2 boundary may be misclassified inliers; tighten and retry. All values are
    // tried first, so exact content keeps an exact map (no corrections).
    let mut found = None;
    for tau in [f64::INFINITY, 0.5, 0.45, 0.4] {
        // With samples clearly off the line, all values together cannot fit exactly: skip that.
        if tau.is_infinite() && clearly_off { continue; }
        let (vi, ks): (Vec<i64>, Vec<i64>) = v.iter().filter(|&&s| res(s) <= tau)
            .map(|&s| (s, ((s as f64 - icpt) / gain).round() as i64)).unzip();
        if vi.len() < MIN_DISTINCT { return None; }
        let kmax = ks.iter().map(|k| k.unsigned_abs()).max().unwrap_or(0);
        // For a given P: (L, U, k of the constraint attaining L, k of the one attaining U). The
        // slack U - L is concave and piecewise linear in P with slope k_L - k_U there.
        let bounds = |p: i128, step: usize| -> (i128, i128, i128, i128) {
            // Exact in i64 whenever nothing can overflow (|s| < 2^25 so |s*2^32| < 2^57; |k*P| <
            // 2^62 checked): true for any gain below ~2^8 on 24-bit content. Otherwise i128.
            if (kmax as u128).saturating_mul(p as u128) < 1 << 62 && p < 1 << 62 {
                let (mut lo_c, mut hi_c, mut kl, mut ku) = (i64::MIN, i64::MAX, 0i128, 0i128);
                let p = p as i64;
                for (&s, &k) in vi.iter().zip(&ks).step_by(step) {
                    let base = (s << SHIFT) - k as i64 * p;
                    if base > lo_c { lo_c = base; kl = k as i128; }
                    let top = base + (1 << SHIFT) - 1;
                    if top < hi_c { hi_c = top; ku = k as i128; }
                }
                return (lo_c as i128, hi_c as i128, kl, ku);
            }
            let (mut lo_c, mut hi_c, mut kl, mut ku) = (i128::MIN, i128::MAX, 0i128, 0i128);
            for (&s, &k) in vi.iter().zip(&ks).step_by(step) {
                let k = k as i128;
                let base = s as i128 * ONE - k * p;
                if base > lo_c { lo_c = base; kl = k; }
                let top = base + ONE - 1;
                if top < hi_c { hi_c = top; ku = k; }
            }
            (lo_c, hi_c, kl, ku)
        };
        let slack = |p: i128, step: usize| { let (a, b, _, _) = bounds(p, step); b - a };
        // Maximum of the slack on [pl, pr]: bisection on the sign of its slope.
        let search = |step: usize, mut pl: i128, mut pr: i128| -> (i128, (i128, i128)) {
            while pr - pl > 2 {
                let m = pl + (pr - pl) / 2;
                let (_, _, kl, ku) = bounds(m, step);
                match (kl - ku).signum() { 1 => pl = m, -1 => pr = m, _ => { pl = m; pr = m; } }
            }
            (pl..=pr).map(|p| { let (a, b, _, _) = bounds(p, step); (p, (a, b)) })
                .max_by_key(|&(_, (a, b))| b - a).expect("non-empty range")
        };
        let p0 = (gain * ONE as f64).round() as i128;
        let (mut pl, mut pr) = ((p0 - P_SEARCH).max(ONE + 1), p0 + P_SEARCH);
        // Feasibility on every 16th value is necessary for feasibility on all, at a sixteenth of
        // the cost. The full set's feasible P lie inside the sub-sample's feasible interval, an
        // interval since the slack is concave: find its ends by bisection and search only there.
        if vi.len() > 16 * MIN_DISTINCT {
            let (ps, (cl, ch)) = search(16, pl, pr);
            if cl > ch { continue; }
            let (mut a, mut b) = (pl, ps); // smallest feasible p in [pl, ps]
            while a < b { let m = a + (b - a) / 2; if slack(m, 16) >= 0 { b = m; } else { a = m + 1; } }
            let (mut c, mut d) = (ps, pr); // largest feasible p in [ps, pr]
            while c < d { let m = c + (d - c + 1) / 2; if slack(m, 16) >= 0 { c = m; } else { d = m - 1; } }
            (pl, pr) = (a, c);
        }
        let (p, (cl, ch)) = search(1, pl, pr);
        if cl <= ch && p <= u64::MAX as i128 { found = Some((p, cl)); break; }
    }
    let (p, cl) = found?;
    // Normalise C into [0, P) by shifting k (C - j*P with k + j gives the same x).
    let c = cl.rem_euclid(p);
    let shift_k = cl.div_euclid(p) as i64; // k under the normalised C = k on the fitted line + this
    let map = ValueMap { p: p as u64, c: c as u64 };
    // A power-of-two step with no offset is exactly FLAC-style wasted bits, which the frame coder
    // already detects per subframe; a map would only add its 17 bytes per channel.
    if p % ONE == 0 && ((p / ONE) as u64).is_power_of_two() && c < ONE { return None; }
    let mut ks = Vec::with_capacity(x.len());
    let mut corr = Vec::new();
    for (i, &s) in x.iter().enumerate() {
        // The map is strictly increasing, so a sample inside the range has at most one exact k:
        // the fitted line's guess, checked in i64, usually is it. Clipped extremes (many k) and
        // misses take the exact i128 path.
        if s != lo && s != hi {
            let g = ((s as f64 - icpt) / gain).round() as i64 + shift_k;
            if let Some(k) = [g, g - 1, g + 1].into_iter().find(|&k| map.raw(k) == s) { ks.push(k); continue; }
        }
        let k = match map.forward(s, lo, hi) {
            Some(k) => k,
            None => {
                let k = map.nearest(s);
                corr.push((i as u32, s - map.raw(k)));
                k
            }
        };
        ks.push(k);
    }
    if corr.len() > budget || corr.iter().any(|&(_, e)| e.abs() > MAX_CORRECTION) { return None; }
    // The k signal must fit the container like the samples do (the frame coder's widths).
    if ks.iter().any(|&k| signed_bits(k) > bits) { return None; }
    Some((ChannelMap { map, corr }, ks))
}

/// Rice parameter minimising the total length of `vals` (0..=24).
fn best_rice(vals: impl Iterator<Item = u64> + Clone) -> u32 {
    (0..=24u32).min_by_key(|&k| vals.clone().map(|v| (v >> k) + 1 + k as u64).sum::<u64>()).expect("non-empty")
}

/// Chunk-payload prefix (mode 0, version 10): `0` = no maps; `1` = one byte per channel: `0` (not
/// mapped), `1` (`P`, `C` as u64 LE), or `2` (`P`, `C`, then corrections: count u32 LE, and if
/// non-zero a u32 LE byte length and that many bytes: Rice parameters for gaps and values (5 bits
/// each), then per correction the gap to the previous position (the first: its position) and
/// `zigzag(e) - 1`, each Rice-coded, zero-padded to a byte).
pub fn write_section(maps: &[Option<ChannelMap>]) -> Vec<u8> {
    if maps.iter().all(Option::is_none) { return vec![0]; }
    let mut out = vec![1];
    for m in maps {
        let Some(m) = m else { out.push(0); continue };
        out.push(if m.corr.is_empty() { 1 } else { 2 });
        out.extend_from_slice(&m.map.p.to_le_bytes());
        out.extend_from_slice(&m.map.c.to_le_bytes());
        if m.corr.is_empty() { continue; }
        let gaps = m.corr.iter().scan(None::<u32>, |prev, &(i, _)| { let g = prev.map_or(i, |p| i - p - 1); *prev = Some(i); Some(g as u64) });
        let vals = m.corr.iter().map(|&(_, e)| crate::rice::zigzag(e) - 1);
        let (kg, kv) = (best_rice(gaps.clone()), best_rice(vals.clone()));
        let mut w = crate::bitio::BitWriter::new();
        w.write_bits(kg as u64, 5);
        w.write_bits(kv as u64, 5);
        for (g, v) in gaps.zip(vals) {
            w.write_unary(g >> kg); if kg > 0 { w.write_bits(g & ((1 << kg) - 1), kg); }
            w.write_unary(v >> kv); if kv > 0 { w.write_bits(v & ((1 << kv) - 1), kv); }
        }
        let bytes = w.finish();
        out.extend_from_slice(&(m.corr.len() as u32).to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&bytes);
    }
    out
}

/// Parses `write_section`'s output from the front of `data` for a chunk of `frames` sample-frames;
/// returns the maps and bytes consumed. Every field is range-checked before use.
pub fn read_section(data: &[u8], nch: usize, frames: usize) -> Result<(Vec<Option<ChannelMap>>, usize), String> {
    const TRUNC: &str = "truncated value-map section";
    let mut maps = vec![None; nch];
    match data.first() {
        None => Err(TRUNC.into()),
        Some(0) => Ok((maps, 1)),
        Some(1) => {
            let mut pos = 1;
            let u32_at = |pos: usize| -> Result<u32, String> { Ok(u32::from_le_bytes(data.get(pos..pos + 4).ok_or(TRUNC)?.try_into().unwrap())) };
            for m in maps.iter_mut() {
                let flag = *data.get(pos).ok_or(TRUNC)?;
                pos += 1;
                if flag == 0 { continue; }
                if flag > 2 { return Err(format!("invalid value-map flag {flag}")); }
                let b = data.get(pos..pos + 16).ok_or(TRUNC)?;
                let vm = ValueMap { p: u64::from_le_bytes(b[..8].try_into().unwrap()), c: u64::from_le_bytes(b[8..].try_into().unwrap()) };
                if !vm.is_valid() { return Err(format!("invalid value map P={} C={}", vm.p, vm.c)); }
                pos += 16;
                let mut corr = Vec::new();
                if flag == 2 {
                    let n = u32_at(pos)? as usize;
                    let len = u32_at(pos + 4)? as usize;
                    pos += 8;
                    if n == 0 || n > frames { return Err(format!("invalid correction count {n} for {frames} frames")); }
                    let body = data.get(pos..pos.checked_add(len).ok_or(TRUNC)?).ok_or(TRUNC)?;
                    pos += len;
                    // each correction costs at least two Rice bits, so a larger count cannot be real and must not size the allocation
                    if n > body.len().saturating_mul(4) { return Err(format!("correction count {n} exceeds its {len}-byte body")); }
                    let mut r = crate::bitio::BitReader::new(body);
                    let e = |x: crate::bitio::BitReaderError| x.0.to_string();
                    let (kg, kv) = (r.read_bits(5).map_err(e)? as u32, r.read_bits(5).map_err(e)? as u32);
                    if kg > 24 || kv > 24 { return Err(format!("invalid correction Rice parameters {kg}/{kv}")); }
                    corr.reserve(n);
                    let mut next = 0u64; // smallest position the next correction may take
                    for _ in 0..n {
                        let g = r.read_rice(kg, (frames as u64 >> kg) + 1).map_err(e)?;
                        let at = next + g;
                        if at >= frames as u64 { return Err(format!("correction position {at} >= {frames}")); }
                        let v = r.read_rice(kv, ((2 * MAX_CORRECTION as u64) >> kv) + 1).map_err(e)?;
                        let val = crate::rice::unzigzag(v + 1);
                        if val.abs() > MAX_CORRECTION { return Err(format!("correction {val} out of range")); }
                        corr.push((at as u32, val));
                        next = at + 1;
                    }
                    if r.byte_pos() != body.len() { return Err("trailing bytes in correction list".into()); }
                }
                *m = Some(ChannelMap { map: vm, corr });
            }
            if maps.iter().all(Option::is_none) { return Err("value-map section maps no channel".into()); }
            Ok((maps, pos))
        }
        Some(t) => Err(format!("invalid value-map section tag {t}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *seed >> 33 }

    /// A smooth-ish 16-bit-range source with every central value in use.
    fn source(n: usize, seed: u64) -> Vec<i64> {
        let mut s = seed;
        let mut acc = 0f64;
        (0..n).map(|i| {
            acc = 0.97 * acc + (lcg(&mut s) % 2001) as f64 - 1000.0;
            ((i as f64 * 0.01).sin() * 2000.0 + acc * 0.5).round().clamp(-32768.0, 32767.0) as i64
        }).collect()
    }

    fn check(x: &[i64], bits: u32, expect: bool) {
        let (lo, hi) = (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1);
        match detect(x, bits) {
            Some((m, ks)) => {
                assert!(expect, "unexpected map {m:?}");
                assert!(m.map.is_valid());
                assert!(m.corr.is_empty(), "exact content needed {} corrections", m.corr.len());
                let mut out = ks.clone();
                m.apply_all(&mut out, lo, hi);
                assert_eq!(out, x);
            }
            None => assert!(!expect, "no map found (len {}, first {:?})", x.len(), &x[..4]),
        }
    }

    #[test]
    fn detects_integer_step_offset_lattice_and_gains() {
        let k = source(40000, 1);
        check(&k.iter().map(|&v| v * 3).collect::<Vec<_>>(), 24, true);
        check(&k.iter().map(|&v| v * 256 + 128).collect::<Vec<_>>(), 24, true);
        // Gains a DAW would apply (10^(dB/20)), rounded to nearest: ties essentially never occur.
        for db in [2.0f64, 0.4, -1.0 + 20.0 * 256f64.log10(), 6.0 + 20.0 * 256f64.log10()] {
            let g = 10f64.powf(db / 20.0);
            let x: Vec<i64> = k.iter().map(|&v| (v as f64 * g).round_ties_even().clamp(-8388608.0, 8388607.0) as i64).collect();
            check(&x, 24, true);
        }
        // An exact .5 tie on every odd k: round-half-up is one floor map, round-half-to-even is not
        // (ties go both ways) -- a known miss that must fall back cleanly, never mis-map.
        check(&k.iter().map(|&v| (v as f64 * 1.5 + 0.5).floor() as i64).collect::<Vec<_>>(), 24, true);
        check(&k.iter().map(|&v| (v as f64 * 1.5).round_ties_even() as i64).collect::<Vec<_>>(), 24, false);
        // A source too sparse for the mean-gap estimate: a large step is still found.
        let sparse: Vec<i64> = source(40000, 5).iter().map(|&v| v * 7 * 37).collect();
        check(&sparse.iter().map(|&v| v * 3).collect::<Vec<_>>(), 24, true);
        // Gain into a 16-bit container with clipping at both ends.
        let peak = k.iter().map(|v| v.abs()).max().unwrap() as f64;
        let g = 40000.0 / peak * 10f64.powf(0.013);
        let x: Vec<i64> = k.iter().map(|&v| (v as f64 * g).round().clamp(-32768.0, 32767.0) as i64).collect();
        assert!(x.contains(&32767) && x.contains(&-32768));
        check(&x, 16, true);
    }

    #[test]
    fn dense_and_random_content_has_no_map() {
        check(&source(40000, 7), 16, false);
        let mut s = 3u64;
        check(&(0..40000).map(|_| (lcg(&mut s) % 65536) as i64 - 32768).collect::<Vec<_>>(), 16, false);
        check(&(0..40000).map(|i| (i % 50) as i64).collect::<Vec<_>>(), 16, false);
        // Gain < 1: dense, not recoverable.
        let k = source(40000, 9);
        check(&k.iter().map(|&v| (v as f64 * 0.9).round() as i64).collect::<Vec<_>>(), 16, false);
    }

    #[test]
    fn apply_matches_the_i128_reference() {
        let mut s = 11u64;
        for i in 0..200_000 {
            let p = (1u64 << 32) + 1 + lcg(&mut s) * lcg(&mut s) * if i % 3 == 0 { 1 << 2 } else { 1 };
            let p = if i % 7 == 0 { u64::MAX - lcg(&mut s) } else { p };
            let c = (lcg(&mut s) * lcg(&mut s) * 4) % p;
            let k = match i % 5 { 0 => i64::MIN + lcg(&mut s) as i64, 1 => i64::MAX - lcg(&mut s) as i64, _ => lcg(&mut s) as i64 % (1 << 27) - (1 << 26) };
            let m = ValueMap { p, c };
            for (lo, hi) in [(-128, 127), (-32768, 32767), (-8388608, 8388607)] {
                assert_eq!(m.apply(k, lo, hi), m.apply_ref(k, lo, hi), "p={p} c={c} k={k}");
            }
        }
    }

    #[test]
    fn clearly_pays_bounds() {
        let m = |g: f64, c: usize| ChannelMap { map: ValueMap { p: (g * 4294967296.0) as u64, c: 0 }, corr: (0..c as u32).map(|i| (i, 1)).collect() };
        for j in 0..33 { assert!(QUARTER_BIT_P[j] as f64 >= 2f64.powf(32.0 + j as f64 / 4.0) * (1.0 - 1e-12)); }
        assert!(m(3.99, 0).clearly_pays(1000) && m(3.99, 55).clearly_pays(1000)); // cellar_44-like
        assert!(!m(1.04, 0).clearly_pays(1000)); // under a quarter-bit: encode both ways
        assert!(m(1.26, 0).clearly_pays(1000) && !m(1.26, 30).clearly_pays(1000));
        assert!(m(3.99, 100).clearly_pays(1000) && !m(3.99, 250).clearly_pays(1000));
    }

    #[test]
    fn apply_is_total_on_hostile_parameters() {
        let m = ValueMap { p: u64::MAX, c: u64::MAX - 1 };
        for k in [i64::MIN, -1, 0, 1, i64::MAX] { let v = m.apply(k, -8388608, 8388607); assert!((-8388608..=8388607).contains(&v)); }
    }

    #[test]
    fn section_roundtrip_and_rejections() {
        let exact = ChannelMap { map: ValueMap { p: 3 << 32, c: 5 }, corr: vec![] };
        let near = ChannelMap { map: ValueMap { p: 3 << 32, c: 5 }, corr: vec![(0, 1), (7, -2), (8, 1), (4000, 300)] };
        let maps = vec![None, Some(exact), Some(near.clone())];
        let b = write_section(&maps);
        assert_eq!(read_section(&b, 3, 4096).unwrap(), (maps, b.len()));
        assert_eq!(read_section(&[0], 3, 1).unwrap().1, 1);
        // A correction position must lie inside the chunk.
        assert!(read_section(&b, 3, 4000).is_err());
        assert!(read_section(&[], 1, 1).is_err());
        assert!(read_section(&[2], 1, 1).is_err());
        assert!(read_section(&[1, 0], 1, 1).is_err()); // maps nothing
        assert!(read_section(&[1, 3], 1, 1).is_err());
        assert!(read_section(&[1, 1, 0, 0], 1, 1).is_err()); // truncated
        let head = |flag: u8, p: u64, c: u64| { let mut v = vec![1, flag]; v.extend_from_slice(&p.to_le_bytes()); v.extend_from_slice(&c.to_le_bytes()); v };
        assert!(read_section(&head(1, 1 << 32, 0), 1, 1).is_err()); // P == 2^32: not > 1
        assert!(read_section(&head(1, 3 << 32, 3 << 32), 1, 1).is_err()); // C == P
        // Corrections: zero count, count above the chunk length, truncated body, trailing bytes,
        // and every truncation of a valid list are rejected, never a panic.
        let one = write_section(&[Some(near)]);
        for cut in 0..one.len() { assert!(read_section(&one[..cut], 1, 4096).is_err(), "cut {cut}"); }
        let mut zero = head(2, 3 << 32, 0); zero.extend_from_slice(&0u32.to_le_bytes()); zero.extend_from_slice(&0u32.to_le_bytes());
        assert!(read_section(&zero, 1, 10).is_err());
        let mut many = head(2, 3 << 32, 0); many.extend_from_slice(&11u32.to_le_bytes()); many.extend_from_slice(&0u32.to_le_bytes());
        assert!(read_section(&many, 1, 10).is_err());
        let mut trail = one.clone();
        let len_at = 1 + 1 + 16 + 4;
        let len = u32::from_le_bytes(trail[len_at..len_at + 4].try_into().unwrap());
        trail[len_at..len_at + 4].copy_from_slice(&(len + 1).to_le_bytes());
        trail.push(0);
        assert!(read_section(&trail, 1, 4096).is_err());
        // Random bytes after a valid correction header never panic.
        let mut st = 5u64;
        for _ in 0..20_000 {
            let mut v = head(2, 3 << 32, 7);
            let n = 1 + lcg(&mut st) % 40;
            v.extend_from_slice(&(n as u32).to_le_bytes());
            let body: Vec<u8> = (0..lcg(&mut st) % 64).map(|_| lcg(&mut st) as u8).collect();
            v.extend_from_slice(&(body.len() as u32).to_le_bytes());
            v.extend_from_slice(&body);
            let _ = read_section(&v, 1, 1 + (lcg(&mut st) % 5000) as usize);
        }
    }

    #[test]
    fn near_map_carries_off_lattice_samples_as_corrections() {
        // A -0.02 dB gain on 22-bit-like material (G ~ 3.99, as measured on cellar_44), with
        // samples nudged off the lattice the way cellar_44's are: rarely (0.2%) in general, often
        // (10%) at large amplitude. A map with corrections must reproduce every sample.
        let (lo, hi) = (-8388608i64, 8388607i64);
        let g = 4.0 * 10f64.powf(-0.02 / 20.0);
        let mut st = 17u64;
        let src = source(60000, 21);
        let peak = src.iter().map(|v| v.abs()).max().unwrap();
        let x: Vec<i64> = src.iter().map(|&k| {
            let v = (k as f64 * g).round() as i64;
            let rate = if k.abs() * 10 > peak * 7 { 10 } else { 500 };
            if lcg(&mut st) % rate == 0 { v + [1, -1, 2][(lcg(&mut st) % 3) as usize] } else { v }
        }).collect();
        let (m, ks) = detect(&x, 24).expect("near-map found");
        assert!(!m.corr.is_empty() && m.corr.len() < x.len() / 20, "{} corrections", m.corr.len());
        let b = write_section(&[Some(m.clone())]);
        let (back, used) = read_section(&b, 1, x.len()).unwrap();
        assert_eq!(used, b.len());
        let mut out = ks.clone();
        back[0].as_ref().unwrap().apply_all(&mut out, lo, hi);
        assert_eq!(out, x);
        // Too many off-lattice samples: no map.
        let noisy: Vec<i64> = x.iter().map(|&v| if lcg(&mut st) % 5 == 0 { v + 1 } else { v }).collect();
        assert!(detect(&noisy, 24).is_none());
    }

}
