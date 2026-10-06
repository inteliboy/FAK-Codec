//! SHA-256 whole-stream integrity hash: a stronger
//! check than the per-chunk CRC-32s already everywhere in this format, matching FLAC's own
//! convention of storing a whole-decoded-PCM hash (FLAC uses MD5; this project uses SHA-256, since
//! MD5 is cryptographically broken and a from-scratch project has no compatibility reason to match
//! FLAC's specific choice, only its role).
//!
//! **Hardware path.** x86-64 SHA extensions
//! (`sha256rnds2`/`sha256msg1`/`sha256msg2`, "SHA-NI") compress blocks when the CPU has them,
//! dispatched at runtime; the portable scalar compression below is the reference, itself checked
//! against the FIPS 180-4 vectors, and every path must match it (differential tests below). An
//! earlier SHA-NI attempt was wrong and removed; this one follows the standard lane
//! layout (state packed as ABEF/CDGH, message words byte-swapped to big-endian lanes, rounds
//! taken two at a time from the low half of `W + K`). AArch64 has the same thing with the ARMv8 SHA2
//! instructions (`sha256h`/`sha256h2`/`sha256su0`/`sha256su1`, `compress_sha2`), runtime-detected.
//! Hashing is also streamed ([`Sha256`]), so neither the padding nor `pcm_digest` copies the whole
//! input first.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Incremental SHA-256: `update` with any number of byte slices, then `finalize`.
pub struct Sha256 { h: [u32; 8], buf: [u8; 64], buf_len: usize, total: u64, hw: bool }

impl Sha256 {
    pub fn new() -> Self { Self::with_hw(sha_ni_available()) }

    fn with_hw(hw: bool) -> Self { Sha256 { h: H0, buf: [0; 64], buf_len: 0, total: 0, hw } }

    fn compress(&mut self, blocks: &[u8]) {
        debug_assert!(blocks.len() % 64 == 0);
        #[cfg(target_arch = "x86_64")]
        if self.hw {
            // Safety: `hw` is only set when `sha_ni_available()` confirmed the features.
            unsafe { compress_sha_ni(&mut self.h, blocks) };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if self.hw {
            // Safety: `hw` is only set when `sha_ni_available()` confirmed the `sha2` feature.
            unsafe { compress_sha2(&mut self.h, blocks) };
            return;
        }
        for b in blocks.chunks_exact(64) { compress_scalar(&mut self.h, b.try_into().unwrap()); }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len < 64 { return; }
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }
        let full = data.len() / 64 * 64;
        if full > 0 { self.compress(&data[..full]); }
        let rest = &data[full..];
        self.buf[..rest.len()].copy_from_slice(rest);
        self.buf_len = rest.len();
    }

    /// Standard padding: `0x80`, zeros, then the bit length as a big-endian `u64`.
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total * 8;
        let mut tail = [0u8; 128];
        tail[..self.buf_len].copy_from_slice(&self.buf[..self.buf_len]);
        tail[self.buf_len] = 0x80;
        let len = if self.buf_len < 56 { 64 } else { 128 };
        tail[len - 8..len].copy_from_slice(&bit_len.to_be_bytes());
        self.compress(&tail[..len]);
        let mut out = [0u8; 32];
        for (i, word) in self.h.iter().enumerate() { out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes()); }
        out
    }
}

impl Default for Sha256 { fn default() -> Self { Self::new() } }

/// The SHA-256 implementation in use on this machine.
pub fn sha256_kernel() -> &'static str {
    if sha_ni_available() { if cfg!(target_arch = "aarch64") { "armv8 sha2 instructions" } else { "sha-ni" } } else { "scalar" }
}

/// Hardware-SHA dispatch gate (x86-64 SHA-NI, AArch64 SHA2). `FAK_DISABLE_SHA_NI=1` forces the
/// scalar compression (timing comparisons).
fn sha_ni_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        return *ON.get_or_init(|| {
            is_x86_feature_detected!("sha") && is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("ssse3")
                && std::env::var_os("FAK_DISABLE_SHA_NI").is_none_or(|v| v != "1")
        });
    }
    #[cfg(target_arch = "aarch64")]
    {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        return *ON.get_or_init(|| {
            std::arch::is_aarch64_feature_detected!("sha2")
                && std::env::var_os("FAK_DISABLE_SHA_NI").is_none_or(|v| v != "1")
        });
    }
    #[allow(unreachable_code)]
    false
}

/// Computes the SHA-256 digest of `data` (SHA-NI when available, else scalar; identical output).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// Digest of the decoded PCM itself, the value stored
/// in `StreamHeader::pcm_hash` and checked by `decoder::verify` -- matches FLAC's own convention of
/// hashing decoded PCM rather than the compressed bytes (so re-encoding with different settings, or
/// a from-scratch reimplementation, still verifies against the same hash). Canonical byte layout:
/// samples interleaved channel-by-channel per frame (frame 0's channels, then frame 1's, ...), each
/// sample truncated to `ceil(bits_per_sample / 8)` little-endian bytes -- an ordinary interleaved-PCM
/// layout, not required to equal any external tool's hash of the same audio, only this codec's own
/// encode-then-decode round trip. `bits_per_sample` must be 8, 16, or 24 (the only values this
/// format's header accepts); other values would silently truncate real sample bits, so the caller is
/// trusted to pass a validated header field here, not stream-declared input. Streamed through a
/// fixed buffer (no copy of the whole interleaved PCM).
pub fn pcm_digest(channels: &[Vec<i64>], bits_per_sample: u8) -> [u8; 32] {
    let mut h = PcmHasher::new(bits_per_sample);
    h.update(channels);
    h.finalize()
}

/// Writes `src`'s samples as `B` little-endian bytes each into channel slot `ci` of interleaved frames.
#[inline(always)]
fn pack<const B: usize>(src: &[i64], out: &mut [u8], stride: usize, ci: usize) {
    for (i, &v) in src.iter().enumerate() {
        let o = i * stride + ci * B;
        out[o..o + B].copy_from_slice(&v.to_le_bytes()[..B]);
    }
}

/// [`pcm_digest`] fed a piece at a time (each piece a whole number of sample-frames), so a
/// streaming encoder can hash audio it does not keep. The digest of the pieces in order equals
/// `pcm_digest` of their concatenation.
pub struct PcmHasher { h: Sha256, bytes_per_sample: usize, buf: Vec<u8> }

impl PcmHasher {
    pub fn new(bits_per_sample: u8) -> Self {
        PcmHasher { h: Sha256::new(), bytes_per_sample: bits_per_sample.div_ceil(8) as usize, buf: Vec::with_capacity(64 * 1024) }
    }
    pub fn update(&mut self, channels: &[Vec<i64>]) {
        let _g = crate::prof::span(crate::prof::Phase::Sha);
        let n = channels.first().map_or(0, |c| c.len());
        let (nch, bps) = (channels.len(), self.bytes_per_sample);
        let stride = nch * bps;
        // Frames are interleaved a block at a time, one channel per pass (fixed-width stores).
        const BLOCK: usize = 4096;
        let mut off = 0;
        while off < n {
            let m = BLOCK.min(n - off);
            self.buf.clear();
            self.buf.resize(m * stride, 0);
            for (ci, c) in channels.iter().enumerate() {
                let src = &c[off..off + m];
                match bps {
                    1 => pack::<1>(src, &mut self.buf, stride, ci),
                    2 => pack::<2>(src, &mut self.buf, stride, ci),
                    3 => pack::<3>(src, &mut self.buf, stride, ci),
                    4 => pack::<4>(src, &mut self.buf, stride, ci),
                    _ => for (i, &v) in src.iter().enumerate() { self.buf[i * stride + ci * bps..][..bps].copy_from_slice(&v.to_le_bytes()[..bps]); },
                }
            }
            self.h.update(&self.buf);
            off += m;
        }
    }
    pub fn finalize(self) -> [u8; 32] {
        self.h.finalize()
    }
}

/// Portable reference implementation: the standard SHA-256 algorithm (FIPS
/// 180-4), padding + big-endian length suffix + the usual message-schedule/compression loop, no
/// hardware acceleration. Every other kernel in this module must match this exactly.
pub fn sha256_scalar(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::with_hw(false);
    h.update(data);
    h.finalize()
}

/// AArch64 SHA2 compression of whole 64-byte blocks: state as two `abcd`/`efgh` vectors, message
/// words byte-swapped to big-endian lanes, four rounds per `sha256h`/`sha256h2` pair on `W + K`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha2,neon")]
unsafe fn compress_sha2(state: &mut [u32; 8], blocks: &[u8]) {
    use std::arch::aarch64::*;
    let mut abcd = vld1q_u32(state.as_ptr());
    let mut efgh = vld1q_u32(state.as_ptr().add(4));
    let kp = K.as_ptr();
    for block in blocks.chunks_exact(64) {
        let (abcd_save, efgh_save) = (abcd, efgh);
        let dp = block.as_ptr();
        let mut w0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(dp)));
        let mut w1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(dp.add(16))));
        let mut w2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(dp.add(32))));
        let mut w3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(dp.add(48))));
        macro_rules! rounds4 { ($w:expr, $i:expr) => {{
            let wk = vaddq_u32($w, vld1q_u32(kp.add(4 * $i)));
            let prev = abcd;
            abcd = vsha256hq_u32(abcd, efgh, wk);
            efgh = vsha256h2q_u32(efgh, prev, wk);
        }}; }
        rounds4!(w0, 0);
        rounds4!(w1, 1);
        rounds4!(w2, 2);
        rounds4!(w3, 3);
        let mut i = 4;
        while i < 16 {
            let w4 = vsha256su1q_u32(vsha256su0q_u32(w0, w1), w2, w3);
            rounds4!(w4, i);
            (w0, w1, w2, w3) = (w1, w2, w3, w4);
            i += 1;
        }
        abcd = vaddq_u32(abcd, abcd_save);
        efgh = vaddq_u32(efgh, efgh_save);
    }
    vst1q_u32(state.as_mut_ptr(), abcd);
    vst1q_u32(state.as_mut_ptr().add(4), efgh);
}

/// SHA-NI compression of whole 64-byte blocks (see the module doc for the layout).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn compress_sha_ni(state: &mut [u32; 8], blocks: &[u8]) {
    use std::arch::x86_64::*;
    // Byte order within each 32-bit lane: message words are big-endian.
    let mask = _mm_set_epi64x(0x0C0D_0E0F_0809_0A0Bu64 as i64, 0x0405_0607_0001_0203u64 as i64);
    let sp = state.as_mut_ptr() as *mut __m128i;
    let dcba = _mm_loadu_si128(sp);
    let efgh = _mm_loadu_si128(sp.add(1));
    let cdab = _mm_shuffle_epi32::<0xB1>(dcba);
    let efgh = _mm_shuffle_epi32::<0x1B>(efgh);
    let mut abef = _mm_alignr_epi8::<8>(cdab, efgh);
    let mut cdgh = _mm_blend_epi16::<0xF0>(efgh, cdab);
    let kp = K.as_ptr() as *const __m128i;

    macro_rules! rounds4 { ($w:expr, $i:expr) => {{
        let t1 = _mm_add_epi32($w, _mm_loadu_si128(kp.add($i)));
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, t1);
        let t2 = _mm_shuffle_epi32::<0x0E>(t1);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, t2);
    }}; }
    // W[i..i+4] from the previous 16 words (v0 = W[i-16..], ..., v3 = W[i-4..]).
    macro_rules! schedule { ($v0:expr, $v1:expr, $v2:expr, $v3:expr) => {{
        let t1 = _mm_sha256msg1_epu32($v0, $v1);
        let t2 = _mm_alignr_epi8::<4>($v3, $v2);
        _mm_sha256msg2_epu32(_mm_add_epi32(t1, t2), $v3)
    }}; }

    for block in blocks.chunks_exact(64) {
        let (abef_save, cdgh_save) = (abef, cdgh);
        let dp = block.as_ptr() as *const __m128i;
        let mut w0 = _mm_shuffle_epi8(_mm_loadu_si128(dp), mask);
        let mut w1 = _mm_shuffle_epi8(_mm_loadu_si128(dp.add(1)), mask);
        let mut w2 = _mm_shuffle_epi8(_mm_loadu_si128(dp.add(2)), mask);
        let mut w3 = _mm_shuffle_epi8(_mm_loadu_si128(dp.add(3)), mask);
        rounds4!(w0, 0);
        rounds4!(w1, 1);
        rounds4!(w2, 2);
        rounds4!(w3, 3);
        let mut i = 4;
        while i < 16 {
            let w4 = schedule!(w0, w1, w2, w3);
            rounds4!(w4, i);
            (w0, w1, w2, w3) = (w1, w2, w3, w4);
            i += 1;
        }
        abef = _mm_add_epi32(abef, abef_save);
        cdgh = _mm_add_epi32(cdgh, cdgh_save);
    }

    let feba = _mm_shuffle_epi32::<0x1B>(abef);
    let dchg = _mm_shuffle_epi32::<0xB1>(cdgh);
    let dcba = _mm_blend_epi16::<0xF0>(feba, dchg);
    let hgef = _mm_alignr_epi8::<8>(dchg, feba);
    _mm_storeu_si128(sp, dcba);
    _mm_storeu_si128(sp.add(1), hgef);
}

fn compress_scalar(h: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 { w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap()); }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
    }
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) = (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g; g = f; f = e; e = d.wrapping_add(t1);
        d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    for (dst, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) { *dst = dst.wrapping_add(v); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }

    /// NIST/FIPS 180-4 standard test vectors.
    #[test]
    fn scalar_matches_known_vectors() {
        assert_eq!(hex(&sha256_scalar(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256_scalar(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            hex(&sha256_scalar(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Exercises multi-block padding (a message just over one 64-byte block).
        assert_eq!(
            hex(&sha256_scalar(&vec![b'a'; 1_000_000])),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn dispatch_matches_scalar_reference_on_this_machine() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for len in [0usize, 1, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 1000, 100_000] {
            let data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let want = sha256_scalar(&data);
            assert_eq!(sha256(&data), want, "len={len}");
            // The hardware path explicitly, when present (the dispatch may be disabled by env).
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("sha") && is_x86_feature_detected!("sse4.1") {
                let mut h = Sha256::with_hw(true);
                h.update(&data);
                assert_eq!(h.finalize(), want, "sha-ni len={len}");
            }
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("sha2") {
                let mut h = Sha256::with_hw(true);
                h.update(&data);
                assert_eq!(h.finalize(), want, "sha2 len={len}");
            }
            // Streaming in uneven pieces gives the one-shot digest.
            let mut h = Sha256::new();
            let mut rest = &data[..];
            let mut step = 1;
            while !rest.is_empty() { let t = step.min(rest.len()); h.update(&rest[..t]); rest = &rest[t..]; step = step * 3 + 1; }
            assert_eq!(h.finalize(), want, "streamed len={len}");
        }
    }

    /// FIPS 180-4 vectors through the hardware path too.
    #[test]
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn sha_ni_matches_known_vectors() {
        #[cfg(target_arch = "x86_64")]
        if !(is_x86_feature_detected!("sha") && is_x86_feature_detected!("sse4.1")) { return; }
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("sha2") { return; }
        let hw = |d: &[u8]| { let mut h = Sha256::with_hw(true); h.update(d); hex(&h.finalize()) };
        assert_eq!(hw(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hw(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(hw(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        assert_eq!(hw(&vec![b'a'; 1_000_000]), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn pcm_digest_is_deterministic_and_sensitive_to_every_field() {
        let base = vec![vec![1i64, -2, 3, -4], vec![10i64, -20, 30, -40]];
        let d0 = pcm_digest(&base, 16);
        assert_eq!(d0, pcm_digest(&base, 16), "same input must hash the same every time");

        let mut sample_changed = base.clone();
        sample_changed[0][1] = -3;
        assert_ne!(d0, pcm_digest(&sample_changed, 16), "changing one sample must change the digest");

        let mut order_changed = base.clone();
        order_changed.swap(0, 1);
        assert_ne!(d0, pcm_digest(&order_changed, 16), "channel order is part of the canonical layout");

        assert_ne!(d0, pcm_digest(&base, 8), "bit depth changes the truncated byte width");

        // Values that fit in 8 bits should hash identically at 8/16/24-bit widths only when the
        // extra bytes are genuinely zero -- here they aren't (negative values sign-extend), so this
        // just documents that bit depth is not merely cosmetic to the digest.
        let empty: Vec<Vec<i64>> = vec![vec![], vec![]];
        assert_eq!(pcm_digest(&empty, 16), sha256(&[]), "no samples must hash like the empty input");
    }
}
