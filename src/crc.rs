//! CRC-32 (reflected, poly 0xEDB88320 — the common "CRC-32"/zip/png variant) for the stream
//! header, and CRC-16 (poly 0x8005, non-reflected, init 0 — the FLAC-style variant) for frames.
//! Both are corruption-detection checks, not cryptographic; used per.

//!
//! Both are table-driven, slicing-by-8 (8 bytes per step, 8 tables of 256 entries, built at compile
//! time). The bit-at-a-time definitions they replace are kept in the tests as the reference: they
//! were ~25% of block-mode decode instructions.

const fn crc32_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 { c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 }; k += 1; }
        t[0][i] = c;
        i += 1;
    }
    let mut s = 1;
    while s < 8 {
        let mut i = 0;
        while i < 256 { t[s][i] = (t[s - 1][i] >> 8) ^ t[0][(t[s - 1][i] & 0xFF) as usize]; i += 1; }
        s += 1;
    }
    t
}

const fn crc16_tables() -> [[u16; 256]; 8] {
    let mut t = [[0u16; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u16) << 8;
        let mut k = 0;
        while k < 8 { c = if c & 0x8000 != 0 { (c << 1) ^ 0x8005 } else { c << 1 }; k += 1; }
        t[0][i] = c;
        i += 1;
    }
    // t[s][v]: the CRC contribution of byte v followed by s zero bytes.
    let mut s = 1;
    while s < 8 {
        let mut i = 0;
        while i < 256 { t[s][i] = (t[s - 1][i] << 8) ^ t[0][(t[s - 1][i] >> 8) as usize]; i += 1; }
        s += 1;
    }
    t
}

static CRC32_T: [[u32; 256]; 8] = crc32_tables();
static CRC16_T: [[u16; 256]; 8] = crc16_tables();

pub fn crc32(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if data.len() >= 64 && clmul_enabled() {
        // Safety: `clmul_enabled()` confirmed pclmulqdq and sse4.1.
        return !unsafe { crc32_clmul(!0, data) };
    }
    #[cfg(target_arch = "aarch64")]
    if crc_insn_enabled() {
        // Safety: `crc_insn_enabled()` confirmed the ARMv8 CRC32 extension.
        return !unsafe { crc32_arm(!0, data) };
    }
    !crc32_update(!0, data)
}

/// The CRC-32 implementation in use on this machine.
pub fn crc32_kernel() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if clmul_enabled() { return "pclmulqdq (carry-less multiply)"; }
    #[cfg(target_arch = "aarch64")]
    if crc_insn_enabled() { return "armv8 crc32 instructions"; }
    "table (slicing)"
}

/// Hardware CRC-32 dispatch gates. `FAK_DISABLE_HW_CRC=1` forces the tables (for
/// timing comparisons); every path is tested against the bit-at-a-time reference.
#[cfg(target_arch = "x86_64")]
fn clmul_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("sse4.1")
        && std::env::var_os("FAK_DISABLE_HW_CRC").is_none_or(|v| v != "1"))
}

#[cfg(target_arch = "aarch64")]
fn crc_insn_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::arch::is_aarch64_feature_detected!("crc")
        && std::env::var_os("FAK_DISABLE_HW_CRC").is_none_or(|v| v != "1"))
}

/// CRC-32 by carry-less multiplication (the folding method of Intel's "Fast CRC Computation for
/// Generic Polynomials Using PCLMULQDQ", reflected form, as in zlib/Linux/crc32fast): four 128-bit
/// accumulators folded 64 bytes per step, then folded to one, reduced to 64 bits and
/// Barrett-reduced to 32. `crc` is the raw register (pre-inverted), `data.len() >= 64`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse4.1")]
unsafe fn crc32_clmul(crc: u32, mut data: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    // x^(4*128+64) mod P, x^(4*128) mod P (fold by 4); x^(128+64), x^128 (fold by 1); x^64; P and
    // mu = floor(x^64 / P), all bit-reflected with the implied x^32 term (33-bit values).
    const K1: i64 = 0x1_5444_2bd4;
    const K2: i64 = 0x1_c6e4_1596;
    const K3: i64 = 0x1_7519_97d0;
    const K4: i64 = 0x0_ccaa_009e;
    const K5: i64 = 0x1_63cd_6124;
    const P_X: i64 = 0x1_DB71_0641;
    const U_PRIME: i64 = 0x1_F701_1641;
    unsafe {
        let get = |d: &mut &[u8]| { let v = _mm_loadu_si128(d.as_ptr() as *const __m128i); *d = &d[16..]; v };
        let fold = |a: __m128i, b: __m128i, k: __m128i| {
            _mm_xor_si128(_mm_xor_si128(b, _mm_clmulepi64_si128(a, k, 0x00)), _mm_clmulepi64_si128(a, k, 0x11))
        };
        let mut x3 = get(&mut data);
        let mut x2 = get(&mut data);
        let mut x1 = get(&mut data);
        let mut x0 = get(&mut data);
        x3 = _mm_xor_si128(x3, _mm_cvtsi32_si128(crc as i32));
        let k1k2 = _mm_set_epi64x(K2, K1);
        while data.len() >= 64 {
            x3 = fold(x3, get(&mut data), k1k2);
            x2 = fold(x2, get(&mut data), k1k2);
            x1 = fold(x1, get(&mut data), k1k2);
            x0 = fold(x0, get(&mut data), k1k2);
        }
        let k3k4 = _mm_set_epi64x(K4, K3);
        let mut x = fold(x3, x2, k3k4);
        x = fold(x, x1, k3k4);
        x = fold(x, x0, k3k4);
        while data.len() >= 16 { x = fold(x, get(&mut data), k3k4); }
        // 128 -> 64 bits.
        let lo32 = _mm_set_epi32(0, 0, 0, !0);
        let x = _mm_xor_si128(_mm_clmulepi64_si128(x, k3k4, 0x10), _mm_srli_si128(x, 8));
        let x = _mm_xor_si128(_mm_clmulepi64_si128(_mm_and_si128(x, lo32), _mm_set_epi64x(0, K5), 0x00), _mm_srli_si128(x, 4));
        // Barrett reduction 64 -> 32 (reflected: the result is the upper half).
        let pu = _mm_set_epi64x(U_PRIME, P_X);
        let t1 = _mm_clmulepi64_si128(_mm_and_si128(x, lo32), pu, 0x10);
        let t2 = _mm_clmulepi64_si128(_mm_and_si128(t1, lo32), pu, 0x00);
        let c = _mm_extract_epi32(_mm_xor_si128(x, t2), 1) as u32;
        crc32_update(c, data)
    }
}

/// CRC-32 with the ARMv8 CRC32 instructions (`crc32x`: this polynomial, reflected), 8 bytes per
/// instruction. `crc` is the raw register (pre-inverted).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32_arm(mut crc: u32, data: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32b, __crc32d};
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks { crc = __crc32d(crc, u64::from_le_bytes(c.try_into().unwrap())); }
    for &b in chunks.remainder() { crc = __crc32b(crc, b); }
    crc
}

/// Table-driven CRC-32 update of the raw (pre-inverted) register.
fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    let t = &CRC32_T;
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let lo = crc ^ u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        crc = t[7][(lo & 0xFF) as usize] ^ t[6][((lo >> 8) & 0xFF) as usize]
            ^ t[5][((lo >> 16) & 0xFF) as usize] ^ t[4][(lo >> 24) as usize]
            ^ t[3][c[4] as usize] ^ t[2][c[5] as usize] ^ t[1][c[6] as usize] ^ t[0][c[7] as usize];
    }
    for &b in chunks.remainder() { crc = (crc >> 8) ^ t[0][((crc ^ b as u32) & 0xFF) as usize]; }
    crc
}

pub fn crc16(data: &[u8]) -> u16 {
    let t = &CRC16_T;
    let mut crc = 0u16;
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let x0 = c[0] ^ (crc >> 8) as u8;
        let x1 = c[1] ^ crc as u8;
        crc = t[7][x0 as usize] ^ t[6][x1 as usize] ^ t[5][c[2] as usize] ^ t[4][c[3] as usize]
            ^ t[3][c[4] as usize] ^ t[2][c[5] as usize] ^ t[1][c[6] as usize] ^ t[0][c[7] as usize];
    }
    for &b in chunks.remainder() { crc = (crc << 8) ^ t[0][((crc >> 8) as u8 ^ b) as usize]; }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bit-at-a-time definitions the reference.
    fn crc32_ref(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    fn crc16_ref(data: &[u8]) -> u16 {
        let mut crc = 0u16;
        for &b in data {
            crc ^= (b as u16) << 8;
            for _ in 0..8 {
                if crc & 0x8000 != 0 { crc = (crc << 1) ^ 0x8005; } else { crc <<= 1; }
            }
        }
        crc
    }

    #[test]
    fn table_driven_matches_bitwise_reference() {
        let mut s = 0x0123_4567_89ab_cdefu64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let data: Vec<u8> = (0..5000).map(|_| next() as u8).collect();
        for len in (0..=64).chain([255, 256, 257, 1000, 4096, 4999, 5000]) {
            for off in 0..3.min(data.len() - len + 1) {
                let d = &data[off..off + len];
                assert_eq!(crc32(d), crc32_ref(d), "crc32 len {len} off {off}");
                assert_eq!(crc16(d), crc16_ref(d), "crc16 len {len} off {off}");
            }
        }
        for d in [&[0u8; 100][..], &[0xFF; 100][..]] {
            assert_eq!(crc32(d), crc32_ref(d));
            assert_eq!(crc16(d), crc16_ref(d));
        }
    }

    /// The hardware paths (whichever this CPU has) and the tables, each against the reference,
    /// over lengths around every folding boundary and unaligned starts.
    #[test]
    fn crc32_every_path_matches_reference() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let data: Vec<u8> = (0..70_000).map(|_| next() as u8).collect();
        let lens = (0..=300).chain([511, 512, 513, 4095, 4096, 4097, 65_536, 69_990]);
        for len in lens {
            for off in [0, 1, 7, 9] {
                let d = &data[off..off + len];
                let want = crc32_ref(d);
                assert_eq!(!crc32_update(!0, d), want, "tables len {len} off {off}");
                assert_eq!(crc32(d), want, "dispatched len {len} off {off}");
                #[cfg(target_arch = "x86_64")]
                if len >= 64 && is_x86_feature_detected!("pclmulqdq") && is_x86_feature_detected!("sse4.1") {
                    assert_eq!(!unsafe { crc32_clmul(!0, d) }, want, "clmul len {len} off {off}");
                }
                #[cfg(target_arch = "aarch64")]
                if std::arch::is_aarch64_feature_detected!("crc") {
                    assert_eq!(!unsafe { crc32_arm(!0, d) }, want, "arm crc len {len} off {off}");
                }
            }
        }
    }

    #[test]
    fn crc32_check_vector() {
        // Standard CRC-32 check value for the ASCII string "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn crc32_empty_and_sensitivity() {
        assert_eq!(crc32(b""), 0);
        let a = crc32(b"hello world");
        let b = crc32(b"hello worle");
        assert_ne!(a, b);
    }

    #[test]
    fn crc16_check_vector() {
        // CRC-16/XMODEM-style (poly 0x8005, init 0) check value for "123456789".
        assert_eq!(crc16(b"123456789"), 0xFEE8);
    }

    #[test]
    fn crc16_sensitivity() {
        let a = crc16(&[1, 2, 3, 4]);
        let b = crc16(&[1, 2, 3, 5]);
        assert_ne!(a, b);
    }
}
