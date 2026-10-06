//! Reed-Solomon erasure coding over GF(2^16), used by the FEC parity blocks (`format`).
//!
//! A parity block covers `count` data chunks and holds `m` parity shards; shard `j` is
//! `sum_i c(j, i) * chunk_i` (symbol-wise, chunks zero-padded to the longest), with the Cauchy
//! coefficients `c(j, i) = 1 / (j + (1024 + i))` (`+` is XOR). Every square submatrix of a Cauchy matrix
//! is invertible, so *any* `e <= m` lost chunks are rebuilt from any `e` intact shards (MDS: no code
//! with the same redundancy recovers more). Symbols are little-endian 16-bit words; `1024 + count` must
//! not exceed 65536.

use std::sync::OnceLock;

const POLY: u32 = 0x1_100B;
const ORDER: usize = 65535;

struct Tables { exp: Vec<u16>, log: Vec<u16> }

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        // `exp` is doubled so `exp[log a + log b]` needs no modulo.
        let mut exp = vec![0u16; 2 * ORDER + 2];
        let mut log = vec![0u16; ORDER + 1];
        let mut x: u32 = 1;
        for i in 0..ORDER {
            exp[i] = x as u16;
            log[x as usize] = i as u16;
            x <<= 1;
            if x & 0x1_0000 != 0 { x ^= POLY; }
        }
        for i in ORDER..2 * ORDER + 2 { exp[i] = exp[i - ORDER]; }
        Tables { exp, log }
    })
}

#[inline]
pub fn mul(a: u16, b: u16) -> u16 {
    if a == 0 || b == 0 { return 0; }
    let t = tables();
    t.exp[t.log[a as usize] as usize + t.log[b as usize] as usize]
}

#[inline]
pub fn inv(a: u16) -> u16 {
    debug_assert!(a != 0);
    let t = tables();
    t.exp[ORDER - t.log[a as usize] as usize]
}

/// Most parity shards a block may hold; data points start above them so the points never collide
/// and a coefficient does not depend on how many shards a block ended up with.
pub const MAX_SHARDS: usize = 1024;

/// Coefficient of data chunk `i` in parity shard `j`.
#[inline]
pub fn coeff(j: usize, i: usize) -> u16 {
    debug_assert!(j < MAX_SHARDS && MAX_SHARDS + i < 65536);
    inv((j ^ (MAX_SHARDS + i)) as u16)
}

/// Whether a block of `count` chunks and `m` shards fits the field.
pub fn fits(count: usize, m: usize) -> bool { m >= 1 && m <= MAX_SHARDS && count >= 1 && MAX_SHARDS + count <= 65536 }

/// `acc[s] ^= c * data_symbol[s]` for every little-endian 16-bit symbol of `data` (an odd final
/// byte is a symbol with a zero high byte). `data` must not be longer than `2 * acc.len()`.
pub fn mac_bytes(acc: &mut [u16], c: u16, data: &[u8]) {
    debug_assert!(data.len() <= 2 * acc.len());
    if c == 0 { return; }
    let t = tables();
    let lc = t.log[c as usize] as usize;
    let pairs = data.chunks_exact(2);
    let tail = pairs.remainder();
    for (a, p) in acc.iter_mut().zip(pairs) {
        let d = u16::from_le_bytes([p[0], p[1]]);
        if d != 0 { *a ^= t.exp[lc + t.log[d as usize] as usize]; }
    }
    if let [b] = tail {
        let d = *b as u16;
        if d != 0 { acc[data.len() / 2] ^= t.exp[lc + t.log[d as usize] as usize]; }
    }
}

/// `acc[s] ^= c * src[s]` over symbols.
pub fn mac_syms(acc: &mut [u16], c: u16, src: &[u16]) {
    if c == 0 { return; }
    let t = tables();
    let lc = t.log[c as usize] as usize;
    for (a, &d) in acc.iter_mut().zip(src) {
        if d != 0 { *a ^= t.exp[lc + t.log[d as usize] as usize]; }
    }
}

/// Symbols as little-endian bytes.
pub fn syms_to_bytes(s: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() * 2);
    for &v in s { out.extend_from_slice(&v.to_le_bytes()); }
    out
}

/// Inverts the `n x n` matrix `a` (row-major) by Gauss-Jordan; `None` if singular.
pub fn invert(mut a: Vec<u16>, n: usize) -> Option<Vec<u16>> {
    let mut b = vec![0u16; n * n];
    for i in 0..n { b[i * n + i] = 1; }
    for col in 0..n {
        let piv = (col..n).find(|&r| a[r * n + col] != 0)?;
        if piv != col {
            for k in 0..n { a.swap(piv * n + k, col * n + k); b.swap(piv * n + k, col * n + k); }
        }
        let ip = inv(a[col * n + col]);
        for k in 0..n { a[col * n + k] = mul(a[col * n + k], ip); b[col * n + k] = mul(b[col * n + k], ip); }
        for r in 0..n {
            if r == col { continue; }
            let f = a[r * n + col];
            if f == 0 { continue; }
            for k in 0..n {
                let (x, y) = (mul(f, a[col * n + k]), mul(f, b[col * n + k]));
                a[r * n + k] ^= x;
                b[r * n + k] ^= y;
            }
        }
    }
    Some(b)
}

/// Streaming encoder for one parity block: feed the group's chunks in order, then take the shards.
pub struct ShardAcc { pub m: usize, pub shards: Vec<Vec<u16>>, next: usize }

impl ShardAcc {
    pub fn new(m: usize) -> Self { ShardAcc { m, shards: vec![Vec::new(); m], next: 0 } }

    /// Adds the group's next chunk (index = number added so far).
    pub fn push(&mut self, payload: &[u8]) {
        let syms = payload.len().div_ceil(2);
        for (j, sh) in self.shards.iter_mut().enumerate() {
            if sh.len() < syms { sh.resize(syms, 0); }
            mac_bytes(sh, coeff(j, self.next), payload);
        }
        self.next += 1;
    }

    pub fn count(&self) -> usize { self.next }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_is_a_field() {
        let t = tables();
        assert_eq!(t.exp[ORDER], 1);
        // Generator really has full order: no earlier repeat of 1.
        assert!(t.exp[..ORDER].iter().skip(1).all(|&v| v != 1));
        for a in [1u16, 2, 3, 255, 4096, 65535, 12345] {
            assert_eq!(mul(a, inv(a)), 1);
            assert_eq!(mul(a, 1), a);
            assert_eq!(mul(a, 0), 0);
        }
        // Distributive: a*(b^c) == a*b ^ a*c.
        for (a, b, c) in [(7u16, 9, 1234), (65535, 300, 41), (2, 32768, 32769)] {
            assert_eq!(mul(a, b ^ c), mul(a, b) ^ mul(a, c));
        }
    }

    #[test]
    fn cauchy_submatrices_invert() {
        for rows in [[0usize, 1, 2], [1, 3, 4], [0, 2, 4]] {
            for cols in [[0usize, 1, 2], [7, 100, 3000], [0, 59990, 59999]] {
                let n = 3;
                let mut a = vec![0u16; n * n];
                for (r, &j) in rows.iter().enumerate() { for (c, &i) in cols.iter().enumerate() { a[r * n + c] = coeff(j, i); } }
                let inv_a = invert(a.clone(), n).expect("Cauchy submatrix is invertible");
                for r in 0..n { for c in 0..n {
                    let mut s = 0u16;
                    for k in 0..n { s ^= mul(a[r * n + k], inv_a[k * n + c]); }
                    assert_eq!(s, (r == c) as u16);
                } }
            }
        }
    }
}
