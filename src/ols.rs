//! Backward-adaptive stereo OLS stage-1 predictor (H173-H177). Strictly sequential scalar f64 (+, *, /, sqrt only,
//! no FMA, no reordered reductions), so encoder and decoder compute identical predictions on IEEE-754 targets.
//! Ch0 is predicted from its own past `n` and ch1's past `m`; ch1 from its own past `n` and ch0's current + past `m-1`.

#[derive(Clone, Copy, Debug)]
pub struct Params { pub n: usize, pub m: usize, pub lam: f64, pub k: usize, pub reg: f64, pub irls: bool }
impl Default for Params { fn default() -> Self { Params { n: 16, m: 16, lam: 0.998, k: 16, reg: 1.0, irls: false } } }

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

struct Ols { n: usize, r: Vec<f64>, b: Vec<f64>, w: Vec<f64>, l: Vec<f64>, ck: Vec<f64>, y: Vec<f64>, wt: Vec<f64>, lam: f64, k: usize, reg: f64, t: usize, irls: bool, es: f64 }
impl Ols {
    fn new(n: usize, p: &Params) -> Self { Ols { n, r: vec![0.0; n * n], b: vec![0.0; n], w: vec![0.0; n], l: vec![0.0; n * n], ck: vec![0.0; n], y: vec![0.0; n], wt: vec![0.0; n], lam: p.lam, k: p.k, reg: p.reg, t: 0, irls: p.irls, es: 0.0 } }
    fn predict(&self, x: &[f64]) -> i64 {
        let mut s = 0f64;
        for i in 0..self.n { s += self.w[i] * x[i]; }
        s.round().clamp(-(1i64 << 40) as f64, (1i64 << 40) as f64) as i64
    }
    fn update(&mut self, x: &[f64], y: f64) {
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
