//! MSB-first bit-level I/O, the substrate for variable-length (Rice) coding.

/// Complete 32-bit words go to `bytes` (big-endian) as they fill; the `nbits` (< 32) newest bits wait
/// in the low end of `acc`, whose higher bits are stale and never read.
pub struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self { BitWriter { bytes: Vec::new(), acc: 0, nbits: 0 } }

    /// Write the low `n` bits of `v` (0..=57 at a time), MSB first.
    #[inline]
    pub fn write_bits(&mut self, v: u64, n: u32) {
        debug_assert!(n <= 57);
        debug_assert!(n == 64 || v < (1u64 << n));
        if n > 32 {
            self.write_bits(v >> 32, n - 32);
            return self.write_bits(v & 0xFFFF_FFFF, 32);
        }
        // `nbits < 32` and `n <= 32`, so the pending bits and the new ones fit the accumulator.
        self.acc = (self.acc << n) | v;
        self.nbits += n;
        if self.nbits >= 32 {
            self.nbits -= 32;
            self.bytes.extend_from_slice(&((self.acc >> self.nbits) as u32).to_be_bytes());
        }
    }

    /// A Rice code for `z` with parameter `k`: `z >> k` zero bits, a one bit, the low `k` bits of `z`.
    /// One accumulator write when it fits in 32 bits (the same bits `write_unary` + `write_bits` give).
    #[inline]
    pub fn write_rice(&mut self, z: u64, k: u32) {
        let q = z >> k;
        if q + 1 + k as u64 <= 32 {
            self.write_bits((1u64 << k) | (z & ((1u64 << k) - 1)), q as u32 + 1 + k);
        } else {
            self.write_unary(q);
            if k > 0 { self.write_bits(z & ((1u64 << k) - 1), k); }
        }
    }

    /// A recursive Golomb-Rice code for `z` with parameter `k` (format v16): see
    /// `BitReader::read_rgr`.
    pub fn write_rgr(&mut self, z: u64, k: u32) {
        if z < 2 << k { return self.write_bits((2 << k) | z, k + 2); }
        self.write_unary((z >> k) - 1);
        if k > 0 { self.write_bits(z & ((1u64 << k) - 1), k); }
    }

    /// Unary code: `q` zero bits followed by a one bit (used by Rice coding).
    pub fn write_unary(&mut self, mut q: u64) {
        while q >= 32 { self.write_bits(0, 32); q -= 32; }
        self.write_bits(1, (q + 1) as u32);
    }

    /// Pad with zero bits to the next byte boundary and return the finished buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let nbytes = self.nbits.div_ceil(8);
            let v = self.acc << (nbytes * 8 - self.nbits);
            for i in (0..nbytes).rev() { self.bytes.push((v >> (8 * i)) as u8); }
            self.nbits = 0;
        }
        self.bytes
    }

    pub fn bit_len(&self) -> u64 { self.bytes.len() as u64 * 8 + self.nbits as u64 }

    /// Write up to 64 bits by splitting into <=32-bit chunks (write_bits itself caps at 57
    /// to keep the up-to-7-bit leftover plus new bits from overflowing the u64 accumulator).
    /// Write `val` (must fit in `bits`-bit two's complement) as `bits` raw bits.
    pub fn write_signed(&mut self, val: i64, bits: u32) {
        // Up to `write_bits`' 57: a 32-bit stream's widened side channel is 33 bits wide.
        debug_assert!((1..=57).contains(&bits));
        let mask = (1u64 << bits) - 1;
        self.write_bits((val as u64) & mask, bits);
    }

    pub fn write_bits64(&mut self, v: u64, n: u32) {
        debug_assert!(n <= 64);
        if n == 0 { return; }
        if n > 32 {
            self.write_bits(v >> 32, n - 32);
            self.write_bits(v & 0xFFFF_FFFF, 32);
        } else {
            self.write_bits(v, n);
        }
    }
}

impl Default for BitWriter { fn default() -> Self { Self::new() } }

#[derive(Debug)]
pub struct BitReaderError(pub &'static str);

pub struct BitReader<'a> {
    data: &'a [u8],
    bytepos: usize,
    bitpos: u32, // 0..8, bits already consumed from data[bytepos]
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self { BitReader { data, bytepos: 0, bitpos: 0 } }

    /// The next 64 bits big-endian from `bytepos`, shifted so the first unconsumed bit is the MSB
    /// (`64 - bitpos >= 57` valid bits). `None` within 8 bytes of the end -- callers then take the
    /// byte-wise path, so every end-of-stream error still comes from exactly the same code.
    #[inline(always)]
    fn peek64(&self) -> Option<u64> {
        let b = self.data.get(self.bytepos..self.bytepos + 8)?;
        Some(u64::from_be_bytes(b.try_into().expect("8 bytes")) << self.bitpos)
    }

    #[inline(always)]
    fn advance(&mut self, bits: u32) {
        let pos = self.bitpos + bits;
        self.bytepos += (pos >> 3) as usize;
        self.bitpos = pos & 7;
    }

    /// `count` Rice codes with parameter `k`, each mapped through `f` and appended to `out`
    ///: the whole run in one loop with the bit position in a local and the output
    /// written into reserved capacity, instead of one `read_rice` + `push` per code. Any code the
    /// 64-bit fast path cannot take goes through `read_rice` itself, so results and errors are
    /// exactly `read_rice`'s; on error, `out` keeps its previous length.
    #[inline(always)]
    pub fn read_rice_into(&mut self, k: u32, max_q: u64, count: usize, out: &mut Vec<i64>, f: impl Fn(u64) -> i64) -> Result<(), BitReaderError> {
        self.read_golomb_into::<false>(k, max_q, count, out, f)
    }

    /// [`read_rice_into`](Self::read_rice_into) generalized (format v16): with `RGR` each
    /// code is a recursive Golomb-Rice code ([`read_rgr`](Self::read_rgr)) instead of a Rice code.
    /// Results and errors exactly `read_rgr`'s, code for code.
    #[inline(always)]
    pub fn read_golomb_into<const RGR: bool>(&mut self, k: u32, max_q: u64, count: usize, out: &mut Vec<i64>, f: impl Fn(u64) -> i64) -> Result<(), BitReaderError> {
        debug_assert!(k < 62);
        out.reserve(count);
        let base = out.len();
        let dst = out.spare_capacity_mut();
        let data = self.data;
        let dst = &mut dst[..count];
        let mut pos = self.bytepos * 8 + self.bitpos as usize; // absolute bit position
        let mut i = 0;
        while i < count {
            // One 64-bit window, then as many whole codes as it holds: the per-code
            // dependency chain is leading-zero count -> shift, with a load only per window.
            let byte = pos >> 3;
            let sh = (pos & 7) as u32;
            let start = i;
            if let Some(b) = data.get(byte..byte + 8) {
                let mut v = u64::from_be_bytes(b.try_into().expect("8 bytes")) << sh;
                let window = 64 - sh;
                let mut avail = window;
                while i < count {
                    let q = v.leading_zeros();
                    if RGR {
                        // One more low bit when the quotient is 0; the stop bit is never part of
                        // the value, so both cases are `rem + (q + (q != 0)) << k`.
                        // `q == 0` exactly when the window's top bit is set, so the extra bit `e`
                        // comes from `v` in parallel with the leading-zero count: the serial chain
                        // is leading-zero count -> add -> shift, as for Rice.
                        let e = (v >> 63) as u32;
                        let len = q + (k + 1 + e);
                        if len >= avail || q as u64 > max_q { break; }
                        let rem = ((v << q) << 1) >> 1 >> (63 - k - e);
                        dst[i].write(f((((q + 1 - e) as u64) << k) + rem));
                        v <<= len;
                        avail -= len;
                        i += 1;
                        continue;
                    }
                    let len = q + 1 + k;
                    // `>=`: a code that would use the window's last bit waits for the next window,
                    // so the shift below is always < 64 (no guard on the serial chain).
                    if len >= avail || q as u64 > max_q { break; }
                    // The k bits after the stop bit (none when k == 0: the shifts reach 64).
                    let rem = ((v << q) << 1) >> 1 >> (63 - k);
                    dst[i].write(f(((q as u64) << k) | rem));
                    v <<= len;
                    avail -= len;
                    i += 1;
                }
                pos += (window - avail) as usize;
            }
            if i == start {
                // Not even one code fits a fresh window (near the end of the data, or a code
                // longer than ~56 bits): `read_rice`'s byte-wise path, results and errors exact.
                self.bytepos = pos >> 3;
                self.bitpos = (pos & 7) as u32;
                let z = if RGR { self.read_rgr(k, max_q)? } else { self.read_rice(k, max_q)? };
                pos = self.bytepos * 8 + self.bitpos as usize;
                dst[i].write(f(z));
                i += 1;
            }
        }
        // Safety: `reserve` made room for `count`, and all `count` slots past `base` were written
        // above (the loop only leaves early by returning an error, before any length is set).
        unsafe { out.set_len(base + count); }
        self.bytepos = pos >> 3;
        self.bitpos = (pos & 7) as u32;
        Ok(())
    }

    /// One Rice code (unary quotient `q`, then `k` remainder bits), returning `(q << k) | rem`.
    /// Fast path: one 64-bit peek when the whole code lies within it. Otherwise (near the end of
    /// the data, or a code longer than ~56 bits) the byte-wise `read_unary` + `read_bits`, which
    /// were the only path before and give identical results and errors.
    #[inline(always)]
    pub fn read_rice(&mut self, k: u32, max_q: u64) -> Result<u64, BitReaderError> {
        if let Some(v) = self.peek64() {
            let q = v.leading_zeros();
            if q + 1 + k <= 64 - self.bitpos && q as u64 <= max_q {
                let rem = if k == 0 { 0 } else { (v << (q + 1)) >> (64 - k) };
                self.advance(q + 1 + k);
                return Ok(((q as u64) << k) | rem);
            }
        }
        let q = self.read_unary(max_q)?;
        let rem = if k > 0 { self.read_bits(k)? } else { 0 };
        Ok((q << k) | rem)
    }

    /// One recursive Golomb-Rice code (format v16, after SRLA): values below `2^(k+1)` are
    /// `1` and `k + 1` raw bits, larger ones `0` and a Rice code of `z - 2^(k+1)` with parameter
    /// `k`. Read as one unary quotient `q` (the leading `0` included), then `k` low bits, or `k + 1`
    /// when `q == 0`; the value is `low + ((q + (q != 0)) << k)`. So the first `3 * 2^k` values
    /// all cost `k + 2` bits and the tail grows one bit per `2^k`. Byte-wise: the reference the
    /// windowed loop is tested against.
    pub fn read_rgr(&mut self, k: u32, max_q: u64) -> Result<u64, BitReaderError> {
        let q = self.read_unary(max_q)?;
        let e = (q == 0) as u32;
        let low = if k + e > 0 { self.read_bits(k + e)? } else { 0 };
        Ok(low + ((q + 1 - e as u64) << k))
    }

    pub fn read_bits(&mut self, n: u32) -> Result<u64, BitReaderError> {
        debug_assert!(n <= 57);
        if n > 0 {
            if let Some(v) = self.peek64() {
                self.advance(n);
                return Ok(v >> (64 - n));
            }
        }
        let mut out = 0u64;
        let mut remaining = n;
        while remaining > 0 {
            if self.bytepos >= self.data.len() { return Err(BitReaderError("unexpected end of stream")); }
            let avail = 8 - self.bitpos;
            let take = avail.min(remaining);
            let byte = self.data[self.bytepos] as u64;
            let shift = avail - take;
            let bits = (byte >> shift) & ((1u64 << take) - 1);
            out = (out << take) | bits;
            self.bitpos += take;
            remaining -= take;
            if self.bitpos == 8 { self.bitpos = 0; self.bytepos += 1; }
        }
        Ok(out)
    }

    /// Read a unary code (count of leading zero bits before the terminating one), bounded by
    /// `max_q` to keep a corrupted/hostile stream from spinning forever.
    ///
    /// Processes a whole zero-run per byte via `leading_zeros()` (a single hardware instruction on
    /// every target this project builds for) instead of one `read_bits(1)` call per zero bit -- real
    /// profiling found Rice decoding a comparably large
    /// share of block-independent decode time as `lpc::reconstruct`, and unlike that module's
    /// attempted fix, this one doesn't touch any hostile-input bound, only how fast an already-bounded
    /// loop finds the terminating one-bit. `bitpos` is always `< 8` (the struct's own invariant,
    /// renormalized after every read), so `0xFFu8 >> self.bitpos` never shifts by the type width.
    pub fn read_unary(&mut self, max_q: u64) -> Result<u64, BitReaderError> {
        let mut q = 0u64;
        loop {
            if self.bytepos >= self.data.len() { return Err(BitReaderError("unexpected end of stream")); }
            let byte = self.data[self.bytepos];
            // Zero out the already-consumed high bits so `leading_zeros` only ever reports a
            // position within the unconsumed span; `masked`'s low `8 - bitpos` bits are byte's real
            // values, the high `bitpos` bits are forced to 0.
            let masked = byte & (0xFFu8 >> self.bitpos);
            if masked == 0 {
                q += (8 - self.bitpos) as u64;
                if q > max_q { return Err(BitReaderError("unary code exceeds sanity bound")); }
                self.bitpos = 0;
                self.bytepos += 1;
                continue;
            }
            let p = masked.leading_zeros(); // 0..=7; always >= bitpos since positions < bitpos are masked to 0
            q += (p - self.bitpos) as u64;
            if q > max_q { return Err(BitReaderError("unary code exceeds sanity bound")); }
            self.bitpos = p + 1;
            if self.bitpos == 8 { self.bitpos = 0; self.bytepos += 1; }
            return Ok(q);
        }
    }

    /// Bit-by-bit reference kept only for differential testing against the batched `read_unary`
    /// above -- not used by any real encode/decode path.
    #[cfg(test)]
    fn read_unary_scalar_reference(&mut self, max_q: u64) -> Result<u64, BitReaderError> {
        let mut q = 0u64;
        loop {
            if q > max_q { return Err(BitReaderError("unary code exceeds sanity bound")); }
            let bit = self.read_bits(1)?;
            if bit == 1 { return Ok(q); }
            q += 1;
        }
    }

    /// Read up to 64 bits by splitting into <=32-bit chunks (mirrors write_bits64).
    pub fn read_bits64(&mut self, n: u32) -> Result<u64, BitReaderError> {
        debug_assert!(n <= 64);
        if n == 0 { return Ok(0); }
        if n > 32 {
            let hi = self.read_bits(n - 32)?;
            let lo = self.read_bits(32)?;
            Ok((hi << 32) | lo)
        } else {
            self.read_bits(n)
        }
    }

    /// Read `bits` raw bits and sign-extend as a two's-complement integer.
    pub fn read_signed(&mut self, bits: u32) -> Result<i64, BitReaderError> {
        debug_assert!((1..=57).contains(&bits)); // `read_bits`' limit (33 for a 32-bit stream's side channel)
        let v = self.read_bits(bits)?;
        let shift = 64 - bits;
        Ok(((v << shift) as i64) >> shift)
    }

    pub fn align_to_byte(&mut self) {
        if self.bitpos != 0 { self.bitpos = 0; self.bytepos += 1; }
    }

    /// Bits consumed so far (research taps).
    pub fn bit_pos(&self) -> u64 { self.bytepos as u64 * 8 + self.bitpos as u64 }
    pub fn byte_pos(&self) -> usize { self.bytepos + if self.bitpos > 0 { 1 } else { 0 } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_various_widths() {
        let mut w = BitWriter::new();
        let vals: Vec<(u64, u32)> = vec![(0, 1), (1, 1), (5, 3), (255, 8), (1000, 12), (0xFFFF_FFFF, 32), (12345, 20)];
        for &(v, n) in &vals { w.write_bits(v, n); }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &(v, n) in &vals { assert_eq!(r.read_bits(n).unwrap(), v); }
    }

    /// Bit-at-a-time reference reader for the 64-bit-peek fast paths.
    struct RefReader<'a> { data: &'a [u8], pos: usize }
    impl RefReader<'_> {
        fn bit(&mut self) -> Option<u64> {
            let b = *self.data.get(self.pos / 8)?;
            let v = (b >> (7 - self.pos % 8)) & 1;
            self.pos += 1;
            Some(v as u64)
        }
        fn bits(&mut self, n: u32) -> Option<u64> { (0..n).try_fold(0u64, |acc, _| Some((acc << 1) | self.bit()?)) }
        fn rice(&mut self, k: u32, max_q: u64) -> Option<u64> {
            let mut q = 0u64;
            while self.bit()? == 0 { q += 1; if q > max_q { return None; } }
            Some((q << k) | self.bits(k)?)
        }
    }

    #[test]
    fn peek_fast_paths_match_bit_at_a_time_reference() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for trial in 0..3000 {
            let len = (next() % 40) as usize;
            // Sparse-ones streams make long unary runs (fallback path); dense ones short codes.
            let density = trial % 4;
            let data: Vec<u8> = (0..len).map(|_| { let mut b = next() as u8; for _ in 0..density { b &= next() as u8; } b }).collect();
            let mut fast = BitReader::new(&data);
            let mut slow = RefReader { data: &data, pos: 0 };
            for _ in 0..200 {
                let op = next() % 3;
                let (a, b) = match op {
                    0 => { let n = 1 + (next() % 57) as u32; (fast.read_bits(n).ok(), slow.bits(n)) }
                    1 => { let k = (next() % 31) as u32; let mq = [3u64, 60, 1 << 32][(next() % 3) as usize]; (fast.read_rice(k, mq).ok(), slow.rice(k, mq)) }
                    _ => { (fast.read_unary(1 << 32).ok(), slow.rice(0, 1 << 32)) }
                };
                assert_eq!(a, b, "trial {trial} op {op}");
                if a.is_none() { break; }
                assert_eq!(fast.bytepos * 8 + fast.bitpos as usize, slow.pos, "trial {trial} position");
            }
        }
    }

    /// `read_rice_into` must equal a `read_rice` loop code for code: same values, same final
    /// position, and an error exactly where the loop would fail (`out` then unchanged).
    #[test]
    fn read_rice_into_matches_read_rice_loop() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for trial in 0..4000 {
            let len = (next() % 120) as usize;
            let density = trial % 4;
            let data: Vec<u8> = (0..len).map(|_| { let mut b = next() as u8; for _ in 0..density { b &= next() as u8; } b }).collect();
            let k = (next() % 31) as u32;
            let mq = [3u64, 60, 1 << 32][(next() % 3) as usize];
            let count = (next() % 90) as usize;
            let mut a = BitReader::new(&data);
            let mut want = vec![7i64];
            let mut ok = true;
            for _ in 0..count {
                match a.read_rice(k, mq) { Ok(z) => want.push(crate::rice::unzigzag(z)), Err(_) => { ok = false; break; } }
            }
            let mut b = BitReader::new(&data);
            let mut got = vec![7i64];
            let r = b.read_rice_into(k, mq, count, &mut got, crate::rice::unzigzag);
            assert_eq!(r.is_ok(), ok, "trial {trial}");
            if ok {
                assert_eq!(got, want, "trial {trial}");
                assert_eq!((b.bytepos, b.bitpos), (a.bytepos, a.bitpos), "trial {trial} position");
            } else {
                assert_eq!(got, vec![7i64], "trial {trial}: out unchanged on error");
            }
        }
    }

    /// The recursive Golomb-Rice windowed loop against a `read_rgr` loop, as above; and
    /// `write_rgr` round-trips at the length the encoder prices.
    #[test]
    fn read_golomb_into_matches_read_rgr_loop() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for trial in 0..4000 {
            let len = (next() % 120) as usize;
            let density = trial % 4;
            let data: Vec<u8> = (0..len).map(|_| { let mut b = next() as u8; for _ in 0..density { b &= next() as u8; } b }).collect();
            let k = (next() % 30) as u32;
            let mq = [3u64, 60, 1 << 32][(next() % 3) as usize];
            let count = (next() % 90) as usize;
            let mut a = BitReader::new(&data);
            let mut want = vec![7i64];
            let mut ok = true;
            for _ in 0..count {
                match a.read_rgr(k, mq) { Ok(z) => want.push(crate::rice::unzigzag(z)), Err(_) => { ok = false; break; } }
            }
            let mut b = BitReader::new(&data);
            let mut got = vec![7i64];
            let r = b.read_golomb_into::<true>(k, mq, count, &mut got, crate::rice::unzigzag);
            // (`k` up to 29 here, so the `k + 1`-bit case is exercised near the window's end too.)
            assert_eq!(r.is_ok(), ok, "trial {trial}");
            if ok {
                assert_eq!(got, want, "trial {trial}");
                assert_eq!((b.bytepos, b.bitpos), (a.bytepos, a.bitpos), "trial {trial} position");
            } else {
                assert_eq!(got, vec![7i64], "trial {trial}: out unchanged on error");
            }
        }
        for k in [0u32, 1, 5, 20] {
            let vals: Vec<u64> = (0..500u64).map(|i| (i * i * 7919) % (100u64 << k)).chain((0..8u64).map(|j| (j << k) | (j & 1))).collect();
            let mut w = BitWriter::new();
            for &z in &vals {
                let before = w.bit_len();
                w.write_rgr(z, k);
                assert_eq!(w.bit_len() - before, ((z >> k) + k as u64).max(k as u64 + 2), "z {z} k {k}");
            }
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            for &z in &vals { assert_eq!(r.read_rgr(k, 1 << 32).unwrap(), z, "k {k}"); }
        }
    }

    #[test]
    fn write_rice_is_unary_then_remainder_bits() {
        // The single-call form must write exactly the bits `write_unary` + `write_bits` do, including
        // the fallback for codes longer than one 32-bit accumulator write.
        let mut st = 0x1234_5678_9abc_def1u64;
        let mut next = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        for k in 0..24u32 {
            let vals: Vec<u64> = (0..200).map(|i| match i % 4 { 0 => next() % (1 << k.max(1)), 1 => next() % (40u64 << k), 2 => (next() % 5000) << k, _ => 0 }).collect();
            let (mut a, mut b) = (BitWriter::new(), BitWriter::new());
            for &z in &vals {
                a.write_rice(z, k);
                b.write_unary(z >> k);
                if k > 0 { b.write_bits(z & ((1u64 << k) - 1), k); }
            }
            assert_eq!(a.bit_len(), b.bit_len(), "k {k}");
            assert_eq!(a.finish(), b.finish(), "k {k}");
        }
    }

    #[test]
    fn mixed_widths_cross_the_word_boundary_and_read_back() {
        // Random widths 1..=57 (so writes straddle the 32-bit words the writer flushes), then every
        // value read back, and the finished length is the bit length rounded up to bytes.
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        for round in 0..50 {
            let items: Vec<(u64, u32)> = (0..300).map(|_| { let n = 1 + (next() % 57) as u32; (next() & ((1u64 << n) - 1), n) }).collect();
            let mut w = BitWriter::new();
            let mut total = 0u64;
            for &(v, n) in &items { w.write_bits(v, n); total += n as u64; }
            assert_eq!(w.bit_len(), total, "round {round}");
            let bytes = w.finish();
            assert_eq!(bytes.len() as u64, total.div_ceil(8), "round {round}");
            let mut r = BitReader::new(&bytes);
            for &(v, n) in &items { assert_eq!(r.read_bits(n).unwrap(), v, "round {round}"); }
        }
    }

    #[test]
    fn unary_roundtrip() {
        let mut w = BitWriter::new();
        for q in [0u64, 1, 7, 31, 32, 33, 100] { w.write_unary(q); }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for q in [0u64, 1, 7, 31, 32, 33, 100] { assert_eq!(r.read_unary(1000).unwrap(), q); }
    }

    #[test]
    fn read_past_end_errors_not_panics() {
        // `read_bits` itself is capped at 57 bits/call by contract (`debug_assert!(n <= 57)`,
        // mirroring `write_bits`'s accumulator-overflow-avoidance comment) -- a 64-bit read goes
        // through `read_bits64`, the actual public "read up to 64 bits" API, which splits into
        // <=32-bit chunks internally.
        let bytes = vec![0xFFu8];
        let mut r = BitReader::new(&bytes);
        assert!(r.read_bits(1).is_ok());
        assert!(r.read_bits64(64).is_err());
    }

    #[test]
    fn wide_roundtrip_incl_zero_and_64_bits() {
        let mut w = BitWriter::new();
        let vals: Vec<(u64, u32)> = vec![(0, 0), (0, 1), (u64::MAX, 64), (0x1234_5678_9ABC, 48), (7, 3)];
        for &(v, n) in &vals { w.write_bits64(v, n); }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &(v, n) in &vals { assert_eq!(r.read_bits64(n).unwrap(), v); }
    }

    #[test]
    fn signed_roundtrip_boundaries() {
        let mut w = BitWriter::new();
        let vals: Vec<(i64, u32)> = vec![(0, 8), (-1, 8), (127, 8), (-128, 8), (8_388_607, 25), (-8_388_608, 25), (-1, 25)];
        for &(v, n) in &vals { w.write_signed(v, n); }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &(v, n) in &vals { assert_eq!(r.read_signed(n).unwrap(), v); }
    }

    #[test]
    fn unary_hostile_all_zero_bounded() {
        let bytes = vec![0u8; 64];
        let mut r = BitReader::new(&bytes);
        assert!(r.read_unary(100).is_err());
    }

    /// Differential test: the batched `read_unary` must match the bit-by-bit reference exactly,
    /// including which side errors and (when both succeed) the returned value, the reader's
    /// resulting `bytepos`/`bitpos`, and the ability to keep reading correctly afterward -- every
    /// starting bit alignment (0..8), a wide range of `max_q`, and both structured and random data,
    /// since a batched leading-zero-count reimplementation is exactly the kind of hand-derived
    /// bit-manipulation change this project has previously gotten wrong (SHA-NI
    /// kernel) when trusted without this scale of verification.
    #[test]
    fn read_unary_matches_scalar_reference() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let mut patterns: Vec<Vec<u8>> = vec![
            vec![0xFFu8], vec![0x00u8], vec![0x01u8], vec![0x80u8],
            vec![0x00, 0x00, 0x01], vec![0x00, 0x00, 0x00, 0x00, 0x80],
            vec![0x00; 20], vec![0xFF; 20],
        ];
        for _ in 0..200 {
            let len = 1 + (next() % 12) as usize;
            patterns.push((0..len).map(|_| next() as u8).collect());
        }
        // Sparse-ones patterns: mostly zero with occasional set bits, closer to real low-k Rice
        // streams than uniform random bytes (which are mostly-ones bit-wise and rarely exercise
        // multi-byte zero runs).
        for _ in 0..100 {
            let len = 1 + (next() % 16) as usize;
            patterns.push((0..len).map(|_| if next() % 8 == 0 { (1u64 << (next() % 8)) as u8 } else { 0 }).collect());
        }

        for bytes in &patterns {
            for start_bit in 0u32..8 {
                for &max_q in &[0u64, 1, 7, 8, 31, 32, 33, 63, 64, 1000] {
                    let mut a = BitReader::new(bytes);
                    a.bitpos = start_bit;
                    let mut b = BitReader::new(bytes);
                    b.bitpos = start_bit;
                    let ra = a.read_unary(max_q);
                    let rb = b.read_unary_scalar_reference(max_q);
                    match (&ra, &rb) {
                        (Ok(qa), Ok(qb)) => {
                            assert_eq!(qa, qb, "bytes={bytes:?} start_bit={start_bit} max_q={max_q}");
                            assert_eq!((a.bytepos, a.bitpos), (b.bytepos, b.bitpos), "post-read position mismatch bytes={bytes:?} start_bit={start_bit} max_q={max_q}");
                            // Confirm both readers agree on what comes next too, not just this call.
                            let next_a = a.read_bits(1);
                            let next_b = b.read_bits(1);
                            assert_eq!(next_a.is_ok(), next_b.is_ok());
                            if let (Ok(na), Ok(nb)) = (next_a, next_b) { assert_eq!(na, nb); }
                        }
                        (Err(_), Err(_)) => {}
                        _ => panic!("Ok/Err mismatch bytes={bytes:?} start_bit={start_bit} max_q={max_q} batched={ra:?} scalar={rb:?}"),
                    }
                }
            }
        }
    }
}
