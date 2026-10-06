//! Backward-adaptive stereo OLS stage-1 predictor (H173-H177). Scalar-exact f64 (+, *, /, sqrt only, no FMA, no
//! reordered reductions): every element of every sum receives its terms in the same order as the plain sequential form
//! (`plain` below), so encoder and decoder compute identical predictions on IEEE-754 targets. The production size
//! (n = m = 16) runs the `fast` engine (H258): fixed-size, 32-byte-aligned, blocked four lanes wide across *independent*
//! elements, which changes speed, never a result. Any other size, and the tests' oracle, use `plain`.
//! Ch0 is predicted from its own past `n` and ch1's past `m`; ch1 from its own past `n` and ch0's current + past `m-1`.

#[derive(Clone, Copy, Debug)]
pub struct Params { pub n: usize, pub m: usize, pub lam: f64, pub k: usize, pub reg: f64, pub irls: bool }
impl Default for Params { fn default() -> Self { Params { n: 16, m: 16, lam: 0.998, k: 16, reg: 1.0, irls: false } } }

/// The plain sequential engine (row-major lower triangle, right-looking Cholesky), exactly as it was before H258.
/// Used for every size other than the production one, and as the tests' oracle for `fast`.
mod plain {
    use super::{Params, IRLS_BA, IRLS_BS, IRLS_W};
    #[inline(always)]
    fn cholesky_solve(r: &[f64], b: &[f64], n: usize, reg: f64, w: &mut [f64], l: &mut [f64], ck: &mut [f64], y: &mut [f64]) -> bool {
        // Right-looking Cholesky: every element still receives its `- l[i][k]*l[j][k]` terms in ascending k, so the result is
        // bit-identical to the left-looking dot-product form, but the inner update runs over contiguous memory (vectorisable).
        for i in 0..n { for j in 0..=i { l[i * n + j] = r[i * n + j] + if i == j { reg } else { 0.0 }; } }
        for k in 0..n {
            let d = l[k * n + k];
            if d <= 1e-9 { return false; }
            let d = d.sqrt();
            l[k * n + k] = d;
            for i in k + 1..n { let v = l[i * n + k] / d; l[i * n + k] = v; ck[i] = v; }
            for i in k + 1..n {
                let lik = ck[i];
                let row = &mut l[i * n + k + 1..i * n + i + 1];
                for (x, c) in row.iter_mut().zip(&ck[k + 1..=i]) { *x -= lik * *c; }
            }
        }
        for i in 0..n { let mut s = b[i]; for k in 0..i { s -= l[i * n + k] * y[k]; } y[i] = s / l[i * n + i]; }
        for i in (0..n).rev() { let mut s = y[i]; for k in i + 1..n { s -= l[k * n + i] * w[k]; } w[i] = s / l[i * n + i]; }
        true
    }

    pub(super) struct Ols { n: usize, r: Vec<f64>, b: Vec<f64>, w: Vec<f64>, l: Vec<f64>, ck: Vec<f64>, y: Vec<f64>, wt: Vec<f64>, lam: f64, k: usize, reg: f64, t: usize, irls: bool, es: f64 }
    impl Ols {
        pub(super) fn new(n: usize, p: &Params) -> Self { Ols { n, r: vec![0.0; n * n], b: vec![0.0; n], w: vec![0.0; n], l: vec![0.0; n * n], ck: vec![0.0; n], y: vec![0.0; n], wt: vec![0.0; n], lam: p.lam, k: p.k, reg: p.reg, t: 0, irls: p.irls, es: 0.0 } }
        pub(super) fn predict(&self, x: &[f64]) -> i64 {
            let mut s = 0f64;
            for i in 0..self.n { s += self.w[i] * x[i]; }
            s.round().clamp(-(1i64 << 40) as f64, (1i64 << 40) as f64) as i64
        }
        pub(super) fn update(&mut self, x: &[f64], y: f64) {
            #[cfg(target_arch = "x86_64")]
            if crate::simd::avx2_enabled() {
                // Safety: AVX2 confirmed at runtime. No FMA is enabled, so every f64 op rounds exactly as in the baseline clone.
                return unsafe { self.update_avx2(x, y) };
            }
            self.update_impl(x, y)
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        unsafe fn update_avx2(&mut self, x: &[f64], y: f64) { self.update_impl(x, y) }
        #[inline(always)]
        fn update_impl(&mut self, x: &[f64], y: f64) {
            let n = self.n;
            let lam = self.lam;
            // IRLS weight from a decayed sum of recent absolute errors (exact bit extraction + table, config 4 only).
            let c = if self.irls {
                self.es = IRLS_BS * self.es + (y - self.predict(x) as f64).abs();
                let bits = (self.es + IRLS_BA).to_bits();
                let e = ((bits >> 52) & 0x7ff) as usize - 1023;
                IRLS_W[e.min(47)][((bits >> 50) & 3) as usize]
            } else { 1.0 };
            for i in 0..n {
                let xi = if self.irls { c * x[i] } else { x[i] };
                // Per-element ops are independent (no reduction), so the slice form vectorises without changing any result.
                for (r, &xj) in self.r[i * n..i * n + i + 1].iter_mut().zip(&x[..=i]) { *r = lam * *r + xi * xj; }
                self.b[i] = lam * self.b[i] + xi * y;
            }
            // Only the lower triangle is ever read (`cholesky_solve`), so it is not mirrored into the upper one.
            self.t += 1;
            if self.t % self.k == 0 { self.wt.copy_from_slice(&self.w); if cholesky_solve(&self.r, &self.b, n, self.reg, &mut self.wt, &mut self.l, &mut self.ck, &mut self.y) { std::mem::swap(&mut self.w, &mut self.wt); } }
        }
    }
    #[cfg(test)]
    impl Ols {
        pub(super) fn weights(&self) -> &[f64] { &self.w }
    }
}

/// The production engine for `n + m = 32` (H258): every size is a constant, so the loops have no bounds checks and
/// unroll, and every column is 32-byte aligned so each 4-lane block is one aligned vector. Per element the
/// arithmetic is the one of `plain`: the same operations in the same order, only different elements run side by side.
/// Blocks start on a multiple of 4 rows, so up to 3 entries above the diagonal per column are computed as well;
/// nothing that produces a result ever reads them.
mod fast {
    use super::{Params, IRLS_BA, IRLS_BS, IRLS_W};
    pub(super) const D: usize = 32;
    const W: usize = 4;

    /// One column (or vector) of `D` values, 32-byte aligned.
    #[repr(C, align(32))]
    #[derive(Clone, Copy)]
    pub(super) struct Col(pub(super) [f64; D]);
    pub(super) type Mat = [Col; D];
    pub(super) const ZERO: Col = Col([0.0; D]);

    pub(super) struct Fast { rc: Box<Mat>, lt: Box<Mat>, b: Col, w: Col, y: Col, wt: Col, s: Col, xw: Col, lam: f64, k: usize, reg: f64, t: usize, irls: bool, es: f64 }

    impl Fast {
        pub(super) fn new(p: &Params) -> Self {
            Fast { rc: Box::new([ZERO; D]), lt: Box::new([ZERO; D]), b: ZERO, w: ZERO, y: ZERO, wt: ZERO, s: ZERO, xw: ZERO, lam: p.lam, k: p.k, reg: p.reg, t: 0, irls: p.irls, es: 0.0 }
        }
        #[cfg(test)]
        pub(super) fn weights(&self) -> &[f64] { &self.w.0 }
        #[inline(always)]
        fn predict_arr(&self, x: &[f64; D]) -> i64 {
            let mut s = 0f64;
            for i in 0..D { s += self.w.0[i] * x[i]; }
            s.round().clamp(-(1i64 << 40) as f64, (1i64 << 40) as f64) as i64
        }
        pub(super) fn predict(&self, x: &[f64]) -> i64 { self.predict_arr(x[..D].try_into().unwrap()) }
        /// One sample's statistics update; true when a refit (factor + solve) is due.
        pub(super) fn stats(&mut self, x: &[f64], y: f64) -> bool {
            let x: &[f64; D] = x[..D].try_into().unwrap();
            #[cfg(target_arch = "x86_64")]
            if crate::simd::avx2_enabled() {
                // Safety: AVX2 confirmed at runtime. No FMA is enabled, so every f64 op rounds exactly as in the baseline clone.
                return unsafe { self.stats_avx2(x, y) };
            }
            self.stats_impl(x, y)
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        unsafe fn stats_avx2(&mut self, x: &[f64; D], y: f64) -> bool { self.stats_impl(x, y) }
        #[inline(always)]
        fn stats_impl(&mut self, x: &[f64; D], y: f64) -> bool {
            let lam = self.lam;
            if self.irls {
                // IRLS weight from a decayed sum of recent absolute errors (exact bit extraction + table, config 4 only).
                self.es = IRLS_BS * self.es + (y - self.predict_arr(x) as f64).abs();
                let bits = (self.es + IRLS_BA).to_bits();
                let e = ((bits >> 52) & 0x7ff) as usize - 1023;
                let c = IRLS_W[e.min(47)][((bits >> 50) & 3) as usize];
                for i in 0..D { self.xw.0[i] = c * x[i]; }
                rank1(&mut self.rc, &mut self.b.0, lam, &self.xw.0, x, y);
            } else {
                rank1(&mut self.rc, &mut self.b.0, lam, x, x, y);
            }
            self.t += 1;
            self.t % self.k == 0
        }
        /// Statistics update followed by the refit when due.
        pub(super) fn update(&mut self, x: &[f64], y: f64) { if self.stats(x, y) { self.refit(); } }
        /// Cholesky factor and solve from the current statistics; the weights change only if the factorisation succeeds.
        pub(super) fn refit(&mut self) {
            #[cfg(target_arch = "x86_64")]
            if crate::simd::avx2_enabled() {
                // Safety: as in `stats`.
                return unsafe { self.refit_avx2() };
            }
            self.refit_impl()
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        unsafe fn refit_avx2(&mut self) { self.refit_impl() }
        #[inline(always)]
        fn refit_impl(&mut self) {
            self.wt.0 = self.w.0;
            if factor(&self.rc, self.reg, &mut self.lt) {
                solve(&self.lt, &self.b.0, &mut self.wt.0, &mut self.y.0, &mut self.s.0);
                std::mem::swap(&mut self.w, &mut self.wt);
            }
        }
    }

    /// Both channels' refits at once: the two factorisations run one after the other, but the two back substitutions, which
    /// are long dependency chains, are interleaved (H259). Each channel's arithmetic is exactly that of `Fast::refit`.
    pub(super) fn refit_both(a: &mut Fast, b: &mut Fast) {
        #[cfg(target_arch = "x86_64")]
        if crate::simd::avx2_enabled() {
            // Safety: AVX2 confirmed at runtime, no FMA (see `Fast::stats`).
            return unsafe { refit_both_avx2(a, b) };
        }
        refit_both_impl(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn refit_both_avx2(a: &mut Fast, b: &mut Fast) { refit_both_impl(a, b) }
    #[inline(always)]
    fn refit_both_impl(a: &mut Fast, b: &mut Fast) {
        a.wt.0 = a.w.0;
        b.wt.0 = b.w.0;
        let ok = [factor(&a.rc, a.reg, &mut a.lt), factor(&b.rc, b.reg, &mut b.lt)];
        match ok {
            [true, true] => {
                solve_pair([&a.lt, &b.lt], [&a.b.0, &b.b.0], [&mut a.wt.0, &mut b.wt.0], [&mut a.y.0, &mut b.y.0]);
                std::mem::swap(&mut a.w, &mut a.wt);
                std::mem::swap(&mut b.w, &mut b.wt);
            }
            [true, false] => { solve(&a.lt, &a.b.0, &mut a.wt.0, &mut a.y.0, &mut a.s.0); std::mem::swap(&mut a.w, &mut a.wt); }
            [false, true] => { solve(&b.lt, &b.b.0, &mut b.wt.0, &mut b.y.0, &mut b.s.0); std::mem::swap(&mut b.w, &mut b.wt); }
            [false, false] => {}
        }
    }

    /// `r[i][j] = lam * r[i][j] + xw[i] * x[j]` for every `j <= i` (column-major: `rc[j].0[i]`) and
    /// `b[i] = lam * b[i] + xw[i] * y`. `xw` is `x` itself except in the IRLS configuration, where it is the row-weighted copy.
    #[inline(always)]
    pub(super) fn rank1(rc: &mut Mat, b: &mut [f64; D], lam: f64, xw: &[f64; D], x: &[f64; D], y: f64) {
        for j in 0..D {
            let xj = x[j];
            let col = &mut rc[j].0;
            let mut i = j & !(W - 1);
            while i < D {
                for t in 0..W { col[i + t] = lam * col[i + t] + xw[i + t] * xj; }
                i += W;
            }
        }
        for i in 0..D { b[i] = lam * b[i] + xw[i] * y; }
    }

    /// Left-looking Cholesky of `rc + reg * I` into `lt` (column-major: `lt[j].0[i] = L[i][j]`, `i >= j`). Every element
    /// still subtracts its `L[i][k] * L[j][k]` terms one at a time in ascending `k`, and the pivot test, `sqrt` and the
    /// divisions are the ones of the right-looking form, so the result is bit-identical to it. Returns false at the
    /// first pivot `<= 1e-9`.
    #[inline(always)]
    pub(super) fn factor(rc: &Mat, reg: f64, lt: &mut Mat) -> bool {
        for j in 0..D {
            let first = j & !(W - 1);
            let mut dj = 0f64;
            let mut i = first;
            while i < D {
                let mut acc = [0f64; W];
                for t in 0..W { acc[t] = rc[j].0[i + t] + if i + t == j { reg } else { 0.0 }; }
                for k in 0..j {
                    let c = lt[k].0[j];
                    for t in 0..W { acc[t] -= lt[k].0[i + t] * c; }
                }
                if i == first {
                    let d = acc[j - first];
                    if d <= 1e-9 { return false; }
                    dj = d.sqrt();
                }
                for t in 0..W { lt[j].0[i + t] = acc[t] / dj; }
                if i == first { lt[j].0[j] = dj; }
                i += W;
            }
        }
        true
    }

    /// Solves `L L^T w = b`. Forward substitution runs column-oriented (each `s[i]` still gets its terms in ascending
    /// `k`, so it equals the dot-product form; entries `<= k` that share a block are touched but already final); back
    /// substitution cannot be reordered without changing the order in which each sum receives its terms, so it stays a
    /// sequential chain.
    #[inline(always)]
    pub(super) fn solve(lt: &Mat, b: &[f64; D], w: &mut [f64; D], y: &mut [f64; D], s: &mut [f64; D]) {
        *s = *b;
        for k in 0..D {
            let yk = s[k] / lt[k].0[k];
            y[k] = yk;
            let mut i = (k + 1) & !(W - 1);
            while i < D {
                for t in 0..W { s[i + t] -= lt[k].0[i + t] * yk; }
                i += W;
            }
        }
        for i in (0..D).rev() {
            let mut acc = y[i];
            for k in i + 1..D { acc -= lt[i].0[k] * w[k]; }
            w[i] = acc / lt[i].0[i];
        }
    }

    /// `solve` for both channels at once (same reasoning as `factor_pair`).
    #[inline(always)]
    pub(super) fn solve_pair(lt: [&Mat; 2], b: [&[f64; D]; 2], w: [&mut [f64; D]; 2], y: [&mut [f64; D]; 2]) {
        let [w0, w1] = w;
        let [y0, y1] = y;
        let (mut s0, mut s1) = (*b[0], *b[1]);
        for k in 0..D {
            let (yk0, yk1) = (s0[k] / lt[0][k].0[k], s1[k] / lt[1][k].0[k]);
            y0[k] = yk0;
            y1[k] = yk1;
            let mut i = (k + 1) & !(W - 1);
            while i < D {
                for t in 0..W {
                    s0[i + t] -= lt[0][k].0[i + t] * yk0;
                    s1[i + t] -= lt[1][k].0[i + t] * yk1;
                }
                i += W;
            }
        }
        for i in (0..D).rev() {
            let (mut a0, mut a1) = (y0[i], y1[i]);
            for k in i + 1..D {
                a0 -= lt[0][i].0[k] * w0[k];
                a1 -= lt[1][i].0[k] * w1[k];
            }
            w0[i] = a0 / lt[0][i].0[i];
            w1[i] = a1 / lt[1][i].0[i];
        }
    }
}

/// H247/H248 IRLS weight table: `(2^e * (1 + (f + 0.5) / 4))^-0.9`, indexed by the binary exponent `e` and top two
/// mantissa bits `f` of `esum + 2`. Literals (generated once) so no libm `powf` is needed and every target agrees.
const IRLS_W: [[f64; 4]; 48] = [
    [0.8994203919269412, 0.7508057183121132, 0.6459991866839309, 0.5679352896179233],
    [0.48198745386564384, 0.40234682220371115, 0.3461823925539329, 0.30434898592517706],
    [0.25829068116431647, 0.2156123233868729, 0.18551455076831347, 0.1630965832322183],
    [0.13841454884616858, 0.11554378320092187, 0.09941478621391013, 0.08740129486931665],
    [0.07417452014112849, 0.06191838029789741, 0.0532750648238939, 0.046837194216121536],
    [0.03974914114181265, 0.03318123842325825, 0.028549400346575122, 0.02509943091024871],
    [0.021301037317202183, 0.017781385398068894, 0.015299244831391832, 0.013450451987183861],
    [0.011414943260536289, 0.00952880849839029, 0.008198662303565656, 0.007207918749491106],
    [0.006117116631500151, 0.005106362039082509, 0.004393554342629172, 0.0038626280179111755],
    [0.0032780816364406315, 0.0027364316617956735, 0.0023544474753205173, 0.0020699311026231796],
    [0.0017566804529823069, 0.0014664174185783458, 0.0012617171614920522, 0.0011092486125350062],
    [0.0009413817458313354, 0.0007858336371166231, 0.0006761374854569001, 0.0005944316131351113],
    [0.0005044739866490555, 0.00042111781911498606, 0.0003623331069693622, 0.0003185480141254263],
    [0.00027034091571517295, 0.00022567145156430048, 0.0001941695043240432, 0.00017070565404163406],
    [0.00014487210965064156, 0.00012093433651933085, 0.00010405286098416776, 9.147889495336236e-05],
    [7.763504129260283e-05, 6.480710629542626e-05, 5.576054755190452e-05, 4.90223259965795e-05],
    [4.160358851016051e-05, 3.47292683556033e-05, 2.988133756131217e-05, 2.6270414037468475e-05],
    [2.2294811055734936e-05, 1.8610954098418533e-05, 1.601301231165167e-05, 1.4077966307599815e-05],
    [1.1947493420898731e-05, 9.973363357583023e-06, 8.5811608254476e-06, 7.544195347482764e-06],
    [6.4025031961731075e-06, 5.344593089444674e-06, 4.5985302252353845e-06, 4.042834184810898e-06],
    [3.4310165097310673e-06, 2.8640965206608313e-06, 2.464291331039164e-06, 2.1665011963574335e-06],
    [1.8386362223268268e-06, 1.5348313224934043e-06, 1.3205810262830078e-06, 1.1609992444045141e-06],
    [9.853007551739363e-07, 8.22495740458957e-07, 7.076818495495354e-07, 6.221640900887231e-07],
    [5.280096010061972e-07, 4.407645538365243e-07, 3.792373131328968e-07, 3.3340948055006646e-07],
    [2.8295333916140924e-07, 2.3619987601431804e-07, 2.0322824410970264e-07, 1.786697167057858e-07],
    [1.5163094002461487e-07, 1.2657637948325438e-07, 1.089073194373135e-07, 9.574673046206933e-08],
    [8.125700880890723e-08, 6.783060225703769e-08, 5.8361987424437816e-08, 5.130940241693061e-08],
    [4.354455284323229e-08, 3.634951972347369e-08, 3.127541467119466e-08, 2.7496027944530882e-08],
    [2.3334948087692828e-08, 1.947922530777934e-08, 1.6760079737202342e-08, 1.473475653805227e-08],
    [1.2504889055025597e-08, 1.0438658377821627e-08, 8.98150434616286e-09, 7.896160517208783e-09],
    [6.70120412056849e-09, 5.593938516915685e-09, 4.8130690059358666e-09, 4.231447649135612e-09],
    [3.591086371732083e-09, 2.99771742674493e-09, 2.579259816958999e-09, 2.2675766492275662e-09],
    [1.9244155374490944e-09, 1.6064369930839003e-09, 1.3821911124014362e-09, 1.2151642384545371e-09],
    [1.0312687519652286e-09, 8.608682692119615e-10, 7.40697877212689e-10, 6.511903916993485e-10],
    [5.526432405496274e-10, 4.6132788284046483e-10, 3.969301642767629e-10, 3.4896429044098786e-10],
    [2.9615417973557586e-10, 2.472194911782312e-10, 2.1270960827600291e-10, 1.8700533293372913e-10],
    [1.5870509532989693e-10, 1.324816450332767e-10, 1.139882566883551e-10, 1.0021367659556758e-10],
    [8.504795477193805e-11, 7.099515570990954e-11, 6.108479427967707e-11, 5.370317957916187e-11],
    [4.557607048377505e-11, 3.8045361929256514e-11, 3.273453073672332e-11, 2.8778821363383336e-11],
    [2.4423611435596867e-11, 2.0388004644182862e-11, 1.7542000676099333e-11, 1.54221885101734e-11],
    [1.308828929798533e-11, 1.0925661165850945e-11, 9.400525402218488e-12, 8.264546189717991e-12],
    [7.013840569789223e-12, 5.8549168491111904e-12, 5.0376168299980445e-12, 4.4288606430225895e-12],
    [3.758624096580262e-12, 3.137572252116992e-12, 2.6995920164090546e-12, 2.3733676532315174e-12],
    [2.0141967811820868e-12, 1.681383338304612e-12, 1.446675541431033e-12, 1.27185623378779e-12],
    [1.0793813291984909e-12, 9.010310211727828e-13, 7.752542271030525e-13, 6.815708797675543e-13],
    [5.784261322960465e-13, 4.828505687074827e-13, 4.1544845366406814e-13, 3.6524479088618966e-13],
    [3.099708893162049e-13, 2.587532129556185e-13, 2.2263331384444352e-13, 1.9572983710071789e-13],
    [1.661092866639415e-13, 1.3866241349591704e-13, 1.1930623882749425e-13, 1.0488902261555052e-13],
];
const IRLS_BS: f64 = 0.7;
const IRLS_BA: f64 = 2.0;

enum Ols { Fast(Box<fast::Fast>), Plain(plain::Ols) }
impl Ols {
    fn new(n: usize, p: &Params) -> Self {
        if n == fast::D { Ols::Fast(Box::new(fast::Fast::new(p))) } else { Ols::Plain(plain::Ols::new(n, p)) }
    }
    #[inline]
    fn predict(&self, x: &[f64]) -> i64 { match self { Ols::Fast(f) => f.predict(x), Ols::Plain(p) => p.predict(x) } }
    #[inline]
    fn update(&mut self, x: &[f64], y: f64) { match self { Ols::Fast(f) => f.update(x, y), Ols::Plain(p) => p.update(x, y) } }
    #[cfg(test)]
    fn weights(&self) -> &[f64] { match self { Ols::Fast(f) => f.weights(), Ols::Plain(p) => p.weights() } }
}

/// A zero-filled `f64` buffer whose slice starts on a 32-byte boundary (the regressor vectors, H258). Never grown.
struct AVec { v: Vec<f64>, off: usize, len: usize }
impl AVec {
    fn new(len: usize) -> Self {
        let v = vec![0.0; len + 4];
        let off = ((32 - v.as_ptr() as usize % 32) % 32) / 8;
        AVec { v, off, len }
    }
    fn as_slice(&self) -> &[f64] { &self.v[self.off..self.off + self.len] }
    fn as_mut_slice(&mut self) -> &mut [f64] { &mut self.v[self.off..self.off + self.len] }
}

/// Extra history slots kept behind each channel's window before it is moved back to the end of its buffer.
const HIST_SLACK: usize = 256;

/// Sample-by-sample predictor state shared by the forward and inverse transforms.
/// The OLS statistics are not updated until `max(n, m)` sample pairs are in the history (H195): a
/// zero-filled history window looks like a step from 0 to the signal and, on material with a DC
/// offset, poisons the covariance for thousands of samples (+3.1% on 1Apollo11 with 1 s chunks).
/// Each channel's history is a window of `hn = max(n, m)` values, newest first, sliding down a longer buffer (H258).
/// `update0` leaves channel 0's refit (when due) to `update1`, which runs both channels' refits together (H259); nothing reads
/// channel 0's weights in between, so every prediction is unchanged. Always call `update0` and `update1` in pairs.
pub struct Stereo { cur0: i64, t: usize, p: Params, o: [Ols; 2], hb: [Vec<f64>; 2], pos: [usize; 2], hn: usize, x: [AVec; 2], refit0: bool }
impl Stereo {
    pub fn new(p: Params) -> Self {
        let d = p.n + p.m;
        let hn = p.n.max(p.m);
        let cap = hn + HIST_SLACK;
        Stereo { cur0: 0, t: 0, o: [Ols::new(d, &p), Ols::new(d, &p)], hb: [vec![0.0; cap], vec![0.0; cap]], pos: [cap - hn; 2], hn, x: [AVec::new(d), AVec::new(d)], refit0: false, p }
    }
    // history of channel c at t-1-i is hb[c][pos[c] + i]
    fn fill(&mut self, c: usize, other_cur: Option<f64>) {
        let (n, m) = (self.p.n, self.p.m);
        let (pc, po) = (self.pos[c], self.pos[1 - c]);
        let (hc, ho) = (&self.hb[c], &self.hb[1 - c]);
        let x = self.x[c].as_mut_slice();
        x[..n].copy_from_slice(&hc[pc..pc + n]);
        match other_cur {
            Some(v) => { x[n] = v; x[n + 1..n + m].copy_from_slice(&ho[po..po + m - 1]); }
            None => x[n..n + m].copy_from_slice(&ho[po..po + m]),
        }
    }
    fn push(&mut self, c: usize, v: f64) {
        let (hn, cap) = (self.hn, self.hb[c].len());
        let mut p = self.pos[c];
        if p == 0 {
            // Out of room: move the newest `hn - 1` values (the ones that stay in the window) to the end.
            self.hb[c].copy_within(0..hn - 1, cap - hn + 1);
            p = cap - hn + 1;
        }
        p -= 1;
        self.hb[c][p] = v;
        self.pos[c] = p;
    }
    /// Prediction for ch0 at the current time (needs only past samples).
    pub fn predict0(&mut self) -> i64 { self.fill(0, None); self.o[0].predict(self.x[0].as_slice()) }
    pub fn update0(&mut self, s0: i64) {
        if self.warm() {
            match &mut self.o[0] {
                Ols::Fast(f) => self.refit0 = f.stats(self.x[0].as_slice(), s0 as f64),
                o => o.update(self.x[0].as_slice(), s0 as f64),
            }
        }
        self.cur0 = s0;
    }
    fn warm(&self) -> bool { self.t >= self.p.n.max(self.p.m) }
    /// Prediction for ch1 given the already-known current ch0 sample.
    pub fn predict1(&mut self) -> i64 { let c = self.cur0 as f64; self.fill(1, Some(c)); self.o[1].predict(self.x[1].as_slice()) }
    pub fn update1(&mut self, s1: i64) {
        if self.warm() {
            let [o0, o1] = &mut self.o;
            match (o0, o1) {
                (Ols::Fast(a), Ols::Fast(b)) => {
                    let refit1 = b.stats(self.x[1].as_slice(), s1 as f64);
                    match (self.refit0, refit1) {
                        (true, true) => fast::refit_both(a, b),
                        (true, false) => a.refit(),
                        (false, true) => b.refit(),
                        (false, false) => {}
                    }
                }
                (_, o1) => o1.update(self.x[1].as_slice(), s1 as f64),
            }
            self.refit0 = false;
        }
        self.t += 1;
        let c0 = self.cur0 as f64;
        self.push(0, c0); self.push(1, s1 as f64);
    }
}

pub fn forward(p: Params, s: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let len = s[0].len();
    let mut st = Stereo::new(p);
    let mut res = vec![vec![0i64; len]; 2];
    for t in 0..len {
        res[0][t] = s[0][t] - st.predict0(); st.update0(s[0][t]);
        res[1][t] = s[1][t] - st.predict1(); st.update1(s[1][t]);
    }
    res
}

pub fn inverse(p: Params, r: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let len = r[0].len();
    let mut st = Stereo::new(p);
    let mut s = vec![vec![0i64; len]; 2];
    for t in 0..len {
        s[0][t] = r[0][t] + st.predict0(); st.update0(s[0][t]);
        s[1][t] = r[1][t] + st.predict1(); st.update1(s[1][t]);
    }
    s
}

impl Stereo {
    /// Forward over one block of both channels, keeping the predictor state for the next block.
    pub fn forward_block(&mut self, s0: &[i64], s1: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let (mut r0, mut r1) = (Vec::with_capacity(s0.len()), Vec::with_capacity(s0.len()));
        for t in 0..s0.len() {
            r0.push(s0[t] - self.predict0()); self.update0(s0[t]);
            r1.push(s1[t] - self.predict1()); self.update1(s1[t]);
        }
        (r0, r1)
    }
    /// Inverse of [`Stereo::forward_block`] (`r0.len() == r1.len()`).
    pub fn inverse_block(&mut self, r0: &[i64], r1: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let (mut s0, mut s1) = (Vec::with_capacity(r0.len()), Vec::with_capacity(r0.len()));
        for t in 0..r0.len() {
            let a = r0[t] + self.predict0(); self.update0(a); s0.push(a);
            let b = r1[t] + self.predict1(); self.update1(b); s1.push(b);
        }
        (s0, s1)
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod reference {
    //! The plain sequential implementation as it was before H258: the old `Stereo` (a `Vec` history shifted by
    //! `insert(0)`) on top of `plain::Ols`. Copied verbatim; only the module wrapper and the accessor are new.
    use super::plain::Ols;
    use super::Params;
    /// Sample-by-sample predictor state shared by the forward and inverse transforms.
    /// The OLS statistics are not updated until `max(n, m)` sample pairs are in the history (H195): a
    /// zero-filled history window looks like a step from 0 to the signal and, on material with a DC
    /// offset, poisons the covariance for thousands of samples (+3.1% on 1Apollo11 with 1 s chunks).
    pub struct Stereo { cur0: i64, t: usize, p: Params, o: [Ols; 2], h: [Vec<f64>; 2], x: [Vec<f64>; 2] }
    impl Stereo {
        pub fn new(p: Params) -> Self {
            let d = p.n + p.m;
            Stereo { cur0: 0, t: 0, o: [Ols::new(d, &p), Ols::new(d, &p)], h: [vec![0.0; p.n.max(p.m)], vec![0.0; p.n.max(p.m)]], x: [vec![0.0; d], vec![0.0; d]], p }
        }
        // h[c][i] = sample c at t-1-i
        fn fill(&mut self, c: usize, other_cur: Option<f64>) {
            let (n, m) = (self.p.n, self.p.m);
            for i in 0..n { self.x[c][i] = self.h[c][i]; }
            let o = 1 - c;
            match other_cur {
                Some(v) => { self.x[c][n] = v; for i in 0..m - 1 { self.x[c][n + 1 + i] = self.h[o][i]; } }
                None => { for i in 0..m { self.x[c][n + i] = self.h[o][i]; } }
            }
        }
        fn push(&mut self, c: usize, v: f64) { self.h[c].pop(); self.h[c].insert(0, v); }
        /// Prediction for ch0 at the current time (needs only past samples).
        pub fn predict0(&mut self) -> i64 { self.fill(0, None); self.o[0].predict(&self.x[0]) }
        pub fn update0(&mut self, s0: i64) { if self.warm() { self.o[0].update(&self.x[0], s0 as f64); } self.cur0 = s0; }
        fn warm(&self) -> bool { self.t >= self.p.n.max(self.p.m) }
        /// Prediction for ch1 given the already-known current ch0 sample.
        pub fn predict1(&mut self) -> i64 { let c = self.cur0 as f64; self.fill(1, Some(c)); self.o[1].predict(&self.x[1]) }
        pub fn update1(&mut self, s1: i64) {
            if self.warm() { self.o[1].update(&self.x[1], s1 as f64); }
            self.t += 1;
            let c0 = self.cur0 as f64;
            self.push(0, c0); self.push(1, s1 as f64);
        }
    }

    pub fn forward(p: Params, s: &[Vec<i64>]) -> Vec<Vec<i64>> {
        let len = s[0].len();
        let mut st = Stereo::new(p);
        let mut res = vec![vec![0i64; len]; 2];
        for t in 0..len {
            res[0][t] = s[0][t] - st.predict0(); st.update0(s[0][t]);
            res[1][t] = s[1][t] - st.predict1(); st.update1(s[1][t]);
        }
        res
    }

    pub fn inverse(p: Params, r: &[Vec<i64>]) -> Vec<Vec<i64>> {
        let len = r[0].len();
        let mut st = Stereo::new(p);
        let mut s = vec![vec![0i64; len]; 2];
        for t in 0..len {
            s[0][t] = r[0][t] + st.predict0(); st.update0(s[0][t]);
            s[1][t] = r[1][t] + st.predict1(); st.update1(s[1][t]);
        }
        s
    }

    impl Stereo {
        /// Forward over one block of both channels, keeping the predictor state for the next block.
        pub fn forward_block(&mut self, s0: &[i64], s1: &[i64]) -> (Vec<i64>, Vec<i64>) {
            let (mut r0, mut r1) = (Vec::with_capacity(s0.len()), Vec::with_capacity(s0.len()));
            for t in 0..s0.len() {
                r0.push(s0[t] - self.predict0()); self.update0(s0[t]);
                r1.push(s1[t] - self.predict1()); self.update1(s1[t]);
            }
            (r0, r1)
        }
        /// Inverse of [`Stereo::forward_block`] (`r0.len() == r1.len()`).
        pub fn inverse_block(&mut self, r0: &[i64], r1: &[i64]) -> (Vec<i64>, Vec<i64>) {
            let (mut s0, mut s1) = (Vec::with_capacity(r0.len()), Vec::with_capacity(r0.len()));
            for t in 0..r0.len() {
                let a = r0[t] + self.predict0(); self.update0(a); s0.push(a);
                let b = r1[t] + self.predict1(); self.update1(b); s1.push(b);
            }
            (s0, s1)
        }
    }

    impl Stereo {
        pub fn weights(&self, c: usize) -> &[f64] { self.o[c].weights() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(seed: &mut u64) -> u64 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *seed >> 33 }

    /// Stereo test signals confined to `bits`-bit samples: noise, tone+noise, impulses, silence, DC, full-scale square.
    fn signal(kind: usize, len: usize, bits: u32, seed: u64) -> [Vec<i64>; 2] {
        let mut s = seed;
        let amp = (1i64 << (bits - 1)) - 1;
        let mut ch = [Vec::with_capacity(len), Vec::with_capacity(len)];
        for t in 0..len {
            for c in 0..2 {
                let r = (next(&mut s) % (2 * amp as u64 + 1)) as i64 - amp;
                let v = match kind {
                    0 => r,
                    1 => ((amp as f64 * 0.7) * ((t as f64) * (0.031 + 0.002 * c as f64)).sin()) as i64 + r / 64,
                    2 => if t % 97 == 0 { amp } else if t % 97 == 1 { -amp } else { r / 1024 },
                    3 => 0,
                    4 => amp / 3,
                    _ => if (t / 7) % 2 == 0 { amp } else { -amp },
                };
                ch[c].push(v.clamp(-amp - 1, amp));
            }
        }
        ch
    }

    fn same_bits(a: &[f64], b: &[f64]) -> bool { a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) }

    /// The `fast` engine (and the ring history for every size) must reproduce the plain sequential implementation
    /// bit for bit: every prediction, and every weight vector after every sample.
    #[test]
    fn production_ols_matches_the_sequential_reference_bit_for_bit() {
        let base = Params::default();
        // (n, m, k, reg, irls). n + m = 32 runs the fast engine, everything else the plain one behind the new history
        // ring. k = 1 solves every sample; reg = 0 lets a pivot fail on degenerate input (the old weights are kept).
        let cases = [
            (16, 16, 16, 1.0, false), (16, 16, 16, 1.0, true), (16, 16, 1, 1.0, false), (16, 16, 7, 1.0, true),
            (16, 16, 16, 0.0, false), (16, 16, 3, 0.0, true), (20, 12, 16, 1.0, false), (12, 20, 4, 1.0, true),
            (5, 3, 1, 1.0, false), (7, 2, 3, 1.0, true), (1, 1, 2, 1.0, false), (3, 5, 4, 1.0, true), (2, 1, 1, 1.0, false),
        ];
        for (ci, &(n, m, k, reg, irls)) in cases.iter().enumerate() {
            let p = Params { n, m, k, reg, irls, ..base };
            for kind in 0..6 {
                let bits = if kind % 2 == 0 { 24 } else { 16 };
                let sig = signal(kind, 1500, bits, 0x9E37 + (ci * 7 + kind) as u64);
                let (mut a, mut b) = (Stereo::new(p), reference::Stereo::new(p));
                for t in 0..sig[0].len() {
                    let (p0a, p0b) = (a.predict0(), b.predict0());
                    assert_eq!(p0a, p0b, "case {ci} kind {kind} t {t}: ch0 prediction");
                    a.update0(sig[0][t]); b.update0(sig[0][t]);
                    let (p1a, p1b) = (a.predict1(), b.predict1());
                    assert_eq!(p1a, p1b, "case {ci} kind {kind} t {t}: ch1 prediction");
                    a.update1(sig[1][t]); b.update1(sig[1][t]);
                    for c in 0..2 {
                        assert!(same_bits(a.o[c].weights(), b.weights(c)), "case {ci} kind {kind} t {t}: ch{c} weights differ");
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn rank1_avx2(rc: &mut fast::Mat, b: &mut [f64; 32], lam: f64, x: &[f64; 32], y: f64) { fast::rank1(rc, b, lam, x, x, y) }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn factor_avx2(rc: &fast::Mat, reg: f64, lt: &mut fast::Mat) -> bool { fast::factor(rc, reg, lt) }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn solve_avx2(lt: &fast::Mat, b: &[f64; 32], w: &mut [f64; 32], y: &mut [f64; 32], s: &mut [f64; 32]) { fast::solve(lt, b, w, y, s) }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn solve_pair_avx2(lt: [&fast::Mat; 2], b: [&[f64; 32]; 2], w: [&mut [f64; 32]; 2], y: [&mut [f64; 32]; 2]) { fast::solve_pair(lt, b, w, y) }

    /// Timing probe for the `fast` kernels (H258): `cargo test --release -- --ignored --nocapture kernel_timings`.
    #[test]
    #[ignore = "timing probe, not a correctness test"]
    #[cfg(target_arch = "x86_64")]
    fn kernel_timings() {
        use std::hint::black_box;
        use std::time::Instant;
        let best = |reps: usize, f: &mut dyn FnMut()| (0..reps).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min);
        let p = Params::default();
        let sig = signal(1, 6000, 24, 5);
        let mut st = Stereo::new(p);
        let mut xs: Vec<([f64; 32], f64)> = Vec::new();
        for t in 0..sig[0].len() {
            let _ = st.predict0();
            if t >= 2000 { let mut a = [0f64; 32]; a.copy_from_slice(st.x[0].as_slice()); xs.push((a, sig[0][t] as f64)); }
            st.update0(sig[0][t]);
            let _ = st.predict1(); st.update1(sig[1][t]);
        }
        let mut rc0: Box<fast::Mat> = Box::new([fast::ZERO; 32]);
        let mut b0 = [0f64; 32];
        for (x, y) in &xs[..2000] { fast::rank1(&mut rc0, &mut b0, p.lam, x, x, *y); }
        let (mut rc, mut b) = (rc0.clone(), b0);
        let t_up = best(5, &mut || { for _ in 0..50 { for (x, y) in &xs { unsafe { rank1_avx2(&mut rc, &mut b, p.lam, black_box(x), *y) } } } black_box(&rc); });
        let mut lt: Box<fast::Mat> = Box::new([fast::ZERO; 32]);
        let reps = 40_000usize;
        let t_fa = best(5, &mut || { for _ in 0..reps { black_box(unsafe { factor_avx2(black_box(&rc0), p.reg, &mut lt) }); } });
        assert!(fast::factor(&rc0, p.reg, &mut lt));
        let (mut w, mut y, mut s) = ([0f64; 32], [0f64; 32], [0f64; 32]);
        let t_so = best(5, &mut || { for _ in 0..reps { unsafe { solve_avx2(black_box(&lt), &b0, &mut w, &mut y, &mut s) } black_box(&w); } });
        let ns_ = |t: f64, c: usize| t * 1e9 / c as f64;
        println!("rank1 {:.1} ns   factor {:.1} ns   solve {:.1} ns   (per call; factor+solve are once per {} samples)", ns_(t_up, 50 * xs.len()), ns_(t_fa, reps), ns_(t_so, reps), p.k);
        // paired back substitution vs two single calls (the second matrix is a different one, so the chains differ)
        let mut rc1: Box<fast::Mat> = Box::new([fast::ZERO; 32]);
        let mut b1 = [0f64; 32];
        for (x, y) in &xs[500..2500] { fast::rank1(&mut rc1, &mut b1, p.lam, x, x, *y * 0.5); }
        let mut lt1: Box<fast::Mat> = Box::new([fast::ZERO; 32]);
        assert!(fast::factor(&rc1, p.reg, &mut lt1));
        let (mut wb, mut yb, mut sb) = ([0f64; 32], [0f64; 32], [0f64; 32]);
        fast::solve(&lt1, &b1, &mut wb, &mut yb, &mut sb);
        let (mut wp0, mut wp1, mut yp0, mut yp1) = ([0f64; 32], [0f64; 32], [0f64; 32], [0f64; 32]);
        unsafe { solve_pair_avx2([&lt, &lt1], [&b0, &b1], [&mut wp0, &mut wp1], [&mut yp0, &mut yp1]) };
        assert!(same_bits(&wp0, &w) && same_bits(&wp1, &wb), "paired solve differs from the single solves");
        let t_ps = best(5, &mut || { for _ in 0..reps { unsafe { solve_pair_avx2([black_box(&lt), black_box(&lt1)], [&b0, &b1], [&mut wp0, &mut wp1], [&mut yp0, &mut yp1]) } black_box(&wp0); } });
        println!("pair: solve {:.1} ns for both channels (2 single solves {:.1} ns)", ns_(t_ps, reps), 2.0 * ns_(t_so, reps));
    }

    /// Forward and inverse over blocks still undo each other, and the block API agrees with the reference's.
    #[test]
    fn production_ols_block_api_roundtrips_and_matches_reference() {
        let p = Params::default();
        for kind in [0, 1, 5] {
            let sig = signal(kind, 4096, 24, 77 + kind as u64);
            let (mut f, mut rf) = (Stereo::new(p), reference::Stereo::new(p));
            let (r0, r1) = f.forward_block(&sig[0], &sig[1]);
            let (q0, q1) = rf.forward_block(&sig[0], &sig[1]);
            assert!(r0 == q0 && r1 == q1, "kind {kind}: residuals differ from reference");
            let mut inv = Stereo::new(p);
            let (s0, s1) = inv.inverse_block(&r0, &r1);
            assert!(s0 == sig[0] && s1 == sig[1], "kind {kind}: inverse is not exact");
        }
    }
}
