//! Reversible stereo decorrelation, the same four modes FLAC uses. `mid = (l+r)>>1` (arithmetic
//! shift = floor division, exact for two's complement) discards the LSB of `l+r`, but that bit is
//! always recoverable from `side = l-r`'s parity: `l+r` and `l-r` have equal parity because their
//! sum `2*l` is even, so `(l+r) mod 2 == side & 1`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoMode { LeftRight = 0, MidSide = 1, LeftSide = 2, SideRight = 3 }

impl StereoMode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v { 0 => Some(Self::LeftRight), 1 => Some(Self::MidSide), 2 => Some(Self::LeftSide), 3 => Some(Self::SideRight), _ => None }
    }
}

pub fn side(l: &[i64], r: &[i64]) -> Vec<i64> { l.iter().zip(r).map(|(&a, &b)| a - b).collect() }
pub fn mid(l: &[i64], r: &[i64]) -> Vec<i64> { l.iter().zip(r).map(|(&a, &b)| (a + b) >> 1).collect() }

/// Encode (l, r) into the two channels that `mode` actually transmits.
pub fn encode(mode: StereoMode, l: &[i64], r: &[i64]) -> (Vec<i64>, Vec<i64>) {
    match mode {
        StereoMode::LeftRight => (l.to_vec(), r.to_vec()),
        StereoMode::MidSide => (mid(l, r), side(l, r)),
        StereoMode::LeftSide => (l.to_vec(), side(l, r)),
        StereoMode::SideRight => (side(l, r), r.to_vec()),
    }
}

/// Reconstruct (l, r) from the two transmitted channels under `mode`.
/// [`decode`] appending straight onto the channels' own buffers (the decoder's form: no
/// intermediate vectors to copy from). Same arithmetic.
pub fn decode_into(mode: StereoMode, a: &[i64], b: &[i64], l: &mut Vec<i64>, r: &mut Vec<i64>) {
    l.reserve(a.len());
    r.reserve(a.len());
    match mode {
        StereoMode::LeftRight => { l.extend_from_slice(a); r.extend_from_slice(b); }
        StereoMode::MidSide => {
            // Two passes, each vectorizable (one fused pass pushing to both vectors was measured
            // slower).
            let l0 = l.len();
            l.extend(a.iter().zip(b).map(|(&mi, &si)| (2 * mi + (si & 1) + si) / 2));
            r.extend(l[l0..].iter().zip(b).map(|(&li, &si)| li - si));
        }
        StereoMode::LeftSide => { l.extend_from_slice(a); r.extend(a.iter().zip(b).map(|(&li, &si)| li - si)); }
        StereoMode::SideRight => { l.extend(b.iter().zip(a).map(|(&ri, &si)| ri + si)); r.extend_from_slice(b); }
    }
}

pub fn decode(mode: StereoMode, a: &[i64], b: &[i64]) -> (Vec<i64>, Vec<i64>) {
    match mode {
        StereoMode::LeftRight => (a.to_vec(), b.to_vec()),
        StereoMode::MidSide => {
            let (m, s) = (a, b);
            let l: Vec<i64> = m.iter().zip(s).map(|(&mi, &si)| {
                let sum = 2 * mi + (si & 1);
                (sum + si) / 2
            }).collect();
            let r: Vec<i64> = l.iter().zip(s).map(|(&li, &si)| li - si).collect();
            (l, r)
        }
        StereoMode::LeftSide => { let l = a.to_vec(); let r: Vec<i64> = a.iter().zip(b).map(|(&li, &si)| li - si).collect(); (l, r) }
        StereoMode::SideRight => { let r = b.to_vec(); let l: Vec<i64> = b.iter().zip(a).map(|(&ri, &si)| ri + si).collect(); (l, r) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(l: &[i64], r: &[i64]) {
        for mode in [StereoMode::LeftRight, StereoMode::MidSide, StereoMode::LeftSide, StereoMode::SideRight] {
            let (a, b) = encode(mode, l, r);
            let (l2, r2) = decode(mode, &a, &b);
            assert_eq!((l2, r2), (l.to_vec(), r.to_vec()), "mode {mode:?} failed");
        }
    }

    #[test]
    fn decode_into_matches_decode() {
        let mut st = 7u64;
        let a: Vec<i64> = (0..500).map(|_| { st = st.wrapping_mul(6364136223846793005).wrapping_add(1); (st >> 20) as i64 % 70000 - 35000 }).collect();
        let b: Vec<i64> = (0..500).map(|_| { st = st.wrapping_mul(6364136223846793005).wrapping_add(1); (st >> 20) as i64 % 140000 - 70000 }).collect();
        for mode in [StereoMode::LeftRight, StereoMode::MidSide, StereoMode::LeftSide, StereoMode::SideRight] {
            let (l, r) = decode(mode, &a, &b);
            let (mut l2, mut r2) = (vec![9i64], vec![-9i64]);
            decode_into(mode, &a, &b, &mut l2, &mut r2);
            assert_eq!((&l2[1..], &r2[1..]), (&l[..], &r[..]), "{mode:?}");
        }
    }

    #[test]
    fn roundtrip_varied_and_extreme_24bit() {
        roundtrip(&[0, 0, 0], &[0, 0, 0]);
        roundtrip(&[1, -1, 100, -100, 12345], &[2, -3, -100, 100, -12345]);
        let lo = -(1i64 << 23);
        let hi = (1i64 << 23) - 1;
        roundtrip(&[lo, hi, lo, hi, 0, lo], &[hi, lo, hi, lo, 0, hi]);
        roundtrip(&(0..500).map(|i: i64| (i * 37) % 1000 - 500).collect::<Vec<_>>(),
                   &(0..500).map(|i: i64| (i * 91) % 700 - 350).collect::<Vec<_>>());
    }

    #[test]
    fn mid_side_parity_recovers_exactly_for_all_sign_combinations() {
        let pairs = [(-5i64, -3i64), (-5, 3), (5, -3), (5, 3), (-1, 0), (1, 0), (0, -1), (0, 1)];
        let l: Vec<i64> = pairs.iter().map(|p| p.0).collect();
        let r: Vec<i64> = pairs.iter().map(|p| p.1).collect();
        roundtrip(&l, &r);
    }
}
