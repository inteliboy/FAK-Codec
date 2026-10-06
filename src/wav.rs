//! Minimal strict WAV reader/writer, for the CLI only — not part of the codec's own bitstream.
//! Semantics deliberately match `tools/benchmark/wavio.py` (8-bit is unsigned with a 128 offset per
//! the WAV convention; 16/24/32-bit are little-endian two's complement) so Rust- and Python-side
//! tools agree on what "the same PCM" means. (b): 32-bit `WAVE_FORMAT_IEEE_FLOAT` is also
//! accepted -- immediately reduced to the integer PCM domain by `floatpcm::map_to_pcm` on read, and
//! reconstructed by `floatpcm::unmap_from_pcm` on write, so every function below this point still
//! only ever handles integer PCM (`Wav::float_info` carries what's needed to invert the mapping).
use crate::floatpcm::{self, FloatInfo};
use std::fs;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
pub struct WavError(pub String);
impl std::fmt::Display for WavError { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "{}", self.0) } }
impl std::error::Error for WavError {}

pub struct Wav {
    pub channels: Vec<Vec<i64>>,
    pub sample_rate: u32,
    pub bits: u8,
    /// `dwChannelMask` from a `WAVE_FORMAT_EXTENSIBLE` fmt chunk (speaker-position bitmask, e.g.
    /// front-left/front-right/LFE/... for 5.1 -- Microsoft's own convention, the same one every
    /// other real-world multichannel WAV producer uses). `None` for a plain (non-extensible) fmt
    /// chunk, which has no mask field at all -- not the same as `Some(0)`, a real, legal (if
    /// unusual) "no defined layout" mask a genuinely extensible file can declare.
    pub channel_mask: Option<u32>,
    /// (b): `Some` iff the source was 32-bit float PCM -- `channels`/`bits` above are already
    /// the grid-mapped integer domain (`floatpcm::map_to_pcm`); this is what inverts it back to
    /// exact float32 bits on write (`write_wav`) or gets threaded into the `.fak` file's metadata
    /// on encode (`main.rs`). `None` for an integer-PCM source.
    pub float_info: Option<FloatInfo>,
}

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
/// `SubFormat` GUID for PCM data inside a `WAVE_FORMAT_EXTENSIBLE` fmt chunk (`KSDATAFORMAT_SUBTYPE_PCM`,
/// `{00000001-0000-0010-8000-00AA00389B71}`), Microsoft's own standard value -- not this project's own.
const KSDATAFORMAT_SUBTYPE_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
/// `SubFormat` GUID for IEEE float data (`KSDATAFORMAT_SUBTYPE_IEEE_FLOAT`,
/// `{00000003-0000-0010-8000-00AA00389B71}`), Microsoft's own standard value.
const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// Reads a WAV file. Streams the data chunk in blocks instead of holding the raw file bytes as well
/// as the decoded samples (the old whole-file `fs::read` made reading the WAV the
/// encoder's peak-memory moment). `-` reads standard input (e.g. foobar2000's converter piping a
/// WAV into `fak encode - out.fak`).
pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<Wav, WavError> {
    let io = |e: std::io::Error| WavError(e.to_string());
    if path.as_ref().as_os_str() == "-" {
        return read_wav_from(std::io::BufReader::with_capacity(1 << 20, std::io::stdin().lock()), None);
    }
    let f = fs::File::open(path).map_err(io)?;
    let len = f.metadata().map_err(io)?.len();
    read_wav_from(std::io::BufReader::with_capacity(1 << 20, f), Some(len))
}

/// Reads until `buf` is full or the source ends; returns the byte count.
fn fill<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<usize, WavError> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(WavError(e.to_string())),
        }
    }
    Ok(n)
}

/// Skips `n` bytes; false if the source ended first.
fn skip<R: Read>(r: &mut R, n: u64) -> Result<bool, WavError> {
    Ok(std::io::copy(&mut r.take(n), &mut std::io::sink()).map_err(|e| WavError(e.to_string()))? == n)
}

/// Sequential WAV parser over any reader (no seeking, so a pipe works). `source_len`, when known,
/// only bounds the up-front sample reservation; no declared size is trusted for allocation. The fmt
/// chunk must precede the data chunk (the RIFF/WAVE convention every writer follows). A data size of
/// 0 or 0xFFFFFFFF, or one past the end of the source, means "to the end of the source" (streamed
/// WAVs), as before.
pub fn read_wav_from<R: Read>(mut r: R, source_len: Option<u64>) -> Result<Wav, WavError> {
    let h = parse_head(&mut r, source_len)?;
    let (nch, frame_bytes) = (h.nch, h.frame_bytes);
    let mut buf = vec![0u8; ((1 << 20) / frame_bytes).max(1) * frame_bytes];
    let (mut left, mut carry) = (h.limit, 0usize);
    if h.is_float {
        let mut raw: Vec<Vec<u32>> = (0..nch).map(|_| Vec::with_capacity(h.reserve)).collect();
        loop {
            let want = (buf.len() - carry).min(left.min(usize::MAX as u64) as usize);
            let got = fill(&mut r, &mut buf[carry..carry + want])?;
            left -= got as u64;
            let have = carry + got;
            let whole = have / frame_bytes * frame_bytes;
            push_frames_float(&buf[..whole], &mut raw);
            carry = have - whole;
            if got < want || left == 0 { break; }
            buf.copy_within(whole..have, 0);
        }
        if carry != 0 { return Err(WavError("data length not a multiple of frame size".into())); }
        let (channels, out_bits, float_info) = floatpcm::map_to_pcm(&raw);
        return Ok(Wav { channels, sample_rate: h.sample_rate, bits: out_bits, channel_mask: h.channel_mask, float_info: Some(float_info) });
    }
    let mut channels: Vec<Vec<i64>> = (0..nch).map(|_| Vec::with_capacity(h.reserve)).collect();
    loop {
        let want = (buf.len() - carry).min(left.min(usize::MAX as u64) as usize);
        let got = fill(&mut r, &mut buf[carry..carry + want])?;
        left -= got as u64;
        let have = carry + got;
        let whole = have / frame_bytes * frame_bytes;
        push_frames(&buf[..whole], h.bits, &mut channels);
        carry = have - whole;
        if got < want || left == 0 { break; }
        buf.copy_within(whole..have, 0);
    }
    if carry != 0 { return Err(WavError("data length not a multiple of frame size".into())); }
    Ok(Wav { channels, sample_rate: h.sample_rate, bits: h.bits as u8, channel_mask: h.channel_mask, float_info: None })
}

/// What the RIFF/fmt/data headers say; the reader is left at the first byte of sample data.
struct Head {
    nch: usize, sample_rate: u32, bits: u16, channel_mask: Option<u32>, is_float: bool,
    /// Bytes of sample data to read (declared size, capped by the source length when known).
    limit: u64, frame_bytes: usize, reserve: usize,
    /// True when the data length is really known: declared (not the 0 / 0xFFFFFFFF "to the end"
    /// placeholders) and no larger than what the source holds.
    exact: bool,
}

fn parse_head<R: Read>(r: &mut R, source_len: Option<u64>) -> Result<Head, WavError> {
    let mut head = [0u8; 12];
    if fill(r, &mut head)? < 12 || &head[0..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return Err(WavError("not a RIFF/WAVE file".into()));
    }
    let mut consumed = 12u64;
    let mut fmt = None::<(u16, u32, u16, Option<u32>, bool)>;
    let missing = |fmt: &Option<_>| WavError(if fmt.is_some() { "missing data chunk" } else { "missing fmt chunk" }.into());
    loop {
        let mut ch = [0u8; 8];
        if fill(r, &mut ch)? < 8 { return Err(missing(&fmt)); }
        consumed += 8;
        let id = [ch[0], ch[1], ch[2], ch[3]];
        let size = u32::from_le_bytes(ch[4..8].try_into().unwrap());
        let padded = size as u64 + (size & 1) as u64;
        if &id == b"fmt " {
            // Bounded read: a real fmt chunk is 16..=40 bytes; anything past 64 is skipped unread.
            let mut body = vec![0u8; (size as usize).min(64)];
            let got = fill(r, &mut body)?;
            if got < 16 { return Err(WavError("truncated fmt chunk".into())); }
            let mut tag = u16::from_le_bytes(body[0..2].try_into().unwrap());
            let nch = u16::from_le_bytes(body[2..4].try_into().unwrap());
            let sr = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
            let mut channel_mask = None;
            if tag == WAVE_FORMAT_EXTENSIBLE {
                if got < 26 { return Err(WavError("truncated extensible fmt chunk".into())); }
                tag = u16::from_le_bytes(body[24..26].try_into().unwrap());
                channel_mask = Some(u32::from_le_bytes(body[20..24].try_into().unwrap()));
            }
            if tag != WAVE_FORMAT_PCM && tag != WAVE_FORMAT_IEEE_FLOAT {
                return Err(WavError(format!("unsupported WAV format tag {tag}")));
            }
            let is_float = tag == WAVE_FORMAT_IEEE_FLOAT;
            if is_float && bits != 32 { return Err(WavError(format!("unsupported float bit depth {bits}"))); }
            fmt = Some((nch, sr, bits, channel_mask, is_float));
            if !skip(r, padded - got as u64)? { return Err(missing(&fmt)); }
            consumed += padded;
        } else if &id == b"data" {
            let (nch, sr, bits, channel_mask, is_float) = fmt.ok_or_else(|| WavError("data chunk before fmt chunk".into()))?;
            if bits != 8 && bits != 16 && bits != 24 && bits != 32 { return Err(WavError(format!("unsupported bits {bits}"))); }
            if nch == 0 { return Err(WavError("channels == 0".into())); }
            let remaining = source_len.map(|l| l.saturating_sub(consumed));
            let declared = !(size == 0 || size == 0xFFFF_FFFF);
            let limit = if declared { size as u64 } else { u64::MAX };
            let limit = remaining.map_or(limit, |rem| limit.min(rem));
            let nch = nch as usize;
            let frame_bytes = nch * (bits / 8) as usize;
            let reserve = remaining.map_or(0, |rem| (rem.min(limit) / frame_bytes as u64) as usize);
            let exact = declared && remaining.is_some_and(|rem| size as u64 <= rem);
            return Ok(Head { nch, sample_rate: sr, bits, channel_mask, is_float, limit, frame_bytes, reserve, exact });
        } else {
            if !skip(r, padded)? { return Err(missing(&fmt)); }
            consumed += padded;
        }
    }
}

/// A WAV file read a chunk of frames at a time, for encoders that must not hold the whole audio.
/// Integer PCM with a known data length only: `open` returns `Ok(None)` for anything else (float,
/// or an unknown or inconsistent length) so the caller falls back to [`read_wav`].
pub struct WavStream<R: Read> {
    r: R, pub channels: usize, pub sample_rate: u32, pub bits: u8, pub channel_mask: Option<u32>,
    pub frames: u64, left: u64, frame_bytes: usize, buf: Vec<u8>,
}

impl<R: Read> WavStream<R> {
    pub fn open(mut r: R, source_len: u64) -> Result<Option<Self>, WavError> {
        let h = parse_head(&mut r, Some(source_len))?;
        if h.is_float || !h.exact || h.limit % h.frame_bytes as u64 != 0 { return Ok(None); }
        Ok(Some(WavStream {
            r, channels: h.nch, sample_rate: h.sample_rate, bits: h.bits as u8, channel_mask: h.channel_mask,
            frames: h.limit / h.frame_bytes as u64, left: h.limit, frame_bytes: h.frame_bytes, buf: Vec::new(),
        }))
    }

    /// The next `max_frames` frames (fewer at the end; `None` once all are read), one `Vec` per channel.
    pub fn next_chunk(&mut self, max_frames: usize) -> Result<Option<Vec<Vec<i64>>>, WavError> {
        if self.left == 0 { return Ok(None); }
        let want = (max_frames as u64 * self.frame_bytes as u64).min(self.left) as usize;
        self.buf.resize(want, 0);
        let got = fill(&mut self.r, &mut self.buf)?;
        if got < want { return Err(WavError("data chunk shorter than declared".into())); }
        self.left -= want as u64;
        let mut channels: Vec<Vec<i64>> = (0..self.channels).map(|_| Vec::with_capacity(want / self.frame_bytes)).collect();
        push_frames(&self.buf, self.bits as u16, &mut channels);
        Ok(Some(channels))
    }
}

/// Deinterleaves whole frames of little-endian PCM into `channels` (8-bit unsigned, 16/24/32 signed).
fn push_frames(b: &[u8], bits: u16, channels: &mut [Vec<i64>]) {
    let nch = channels.len();
    match bits {
        8 => for f in b.chunks_exact(nch) {
            for (c, &x) in channels.iter_mut().zip(f) { c.push(x as i64 - 128); }
        },
        16 => for f in b.chunks_exact(nch * 2) {
            for (c, x) in channels.iter_mut().zip(f.chunks_exact(2)) { c.push(i16::from_le_bytes([x[0], x[1]]) as i64); }
        },
        24 => for f in b.chunks_exact(nch * 3) {
            for (c, x) in channels.iter_mut().zip(f.chunks_exact(3)) {
                c.push(((((x[0] as u32) | ((x[1] as u32) << 8) | ((x[2] as u32) << 16)) << 8) as i32 >> 8) as i64);
            }
        },
        _ => for f in b.chunks_exact(nch * 4) {
            for (c, x) in channels.iter_mut().zip(f.chunks_exact(4)) {
                c.push(i32::from_le_bytes([x[0], x[1], x[2], x[3]]) as i64);
            }
        },
    }
}

/// Deinterleaves whole frames of little-endian 32-bit float PCM into raw bit patterns ((b)) --
/// never interpreted as a number here, so a `NaN`/`Inf`/`-0.0` payload survives unchanged into
/// `floatpcm::map_to_pcm`.
fn push_frames_float(b: &[u8], channels: &mut [Vec<u32>]) {
    let nch = channels.len();
    for f in b.chunks_exact(nch * 4) {
        for (c, x) in channels.iter_mut().zip(f.chunks_exact(4)) {
            c.push(u32::from_le_bytes([x[0], x[1], x[2], x[3]]));
        }
    }
}

/// A complete WAV header (RIFF + fmt + data chunk header) for `frames` sample-frames. Sizes that do
/// not fit the 32-bit RIFF fields are written as 0xFFFFFFFF ("to end of file", which [`read_wav`]
/// and most readers accept) instead of silently wrapping.
pub fn header_bytes(nch: u16, sample_rate: u32, bits: u8, channel_mask: Option<u32>, frames: u64) -> Result<Vec<u8>, WavError> {
    if nch == 0 { return Err(WavError("channels == 0".into())); }
    if bits != 8 && bits != 16 && bits != 24 && bits != 32 { return Err(WavError(format!("unsupported bits {bits}"))); }
    Ok(header_bytes_impl(nch, sample_rate, bits, channel_mask, frames, WAVE_FORMAT_PCM, &KSDATAFORMAT_SUBTYPE_PCM))
}

/// As `header_bytes`, but for 32-bit `WAVE_FORMAT_IEEE_FLOAT` data ((b)).
pub fn header_bytes_float(nch: u16, sample_rate: u32, channel_mask: Option<u32>, frames: u64) -> Result<Vec<u8>, WavError> {
    if nch == 0 { return Err(WavError("channels == 0".into())); }
    Ok(header_bytes_impl(nch, sample_rate, 32, channel_mask, frames, WAVE_FORMAT_IEEE_FLOAT, &KSDATAFORMAT_SUBTYPE_IEEE_FLOAT))
}

fn header_bytes_impl(nch: u16, sample_rate: u32, bits: u8, channel_mask: Option<u32>, frames: u64, tag: u16, subtype_guid: &[u8; 16]) -> Vec<u8> {
    let bps = (bits / 8) as u64;
    let data_len = frames.saturating_mul(nch as u64 * bps);
    let byte_rate = sample_rate.wrapping_mul(nch as u32 * bps as u32);
    let block_align = nch as u32 * bps as u32;
    let mut fmt = Vec::with_capacity(40);
    fmt.extend_from_slice(&(if channel_mask.is_some() { WAVE_FORMAT_EXTENSIBLE } else { tag }).to_le_bytes());
    fmt.extend_from_slice(&nch.to_le_bytes());
    fmt.extend_from_slice(&sample_rate.to_le_bytes());
    fmt.extend_from_slice(&byte_rate.to_le_bytes());
    fmt.extend_from_slice(&(block_align as u16).to_le_bytes());
    fmt.extend_from_slice(&(bits as u16).to_le_bytes());
    if let Some(mask) = channel_mask {
        fmt.extend_from_slice(&22u16.to_le_bytes()); // cbSize: 2 (validBits) + 4 (mask) + 16 (SubFormat GUID)
        fmt.extend_from_slice(&(bits as u16).to_le_bytes()); // wValidBitsPerSample: no partial-bit packing here
        fmt.extend_from_slice(&mask.to_le_bytes());
        fmt.extend_from_slice(subtype_guid);
    }
    let riff_size = 4 + 8 + fmt.len() as u64 + 8 + data_len.saturating_add(data_len & 1);
    let clamp = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(28 + fmt.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&clamp(riff_size).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    out.extend_from_slice(&fmt);
    out.extend_from_slice(b"data");
    out.extend_from_slice(&clamp(data_len).to_le_bytes());
    out
}

/// Appends `channels` (equal lengths; `Vec`s or slices) to `out` as interleaved little-endian PCM.
///
/// Frame-major, one pass over the output (the per-channel strided form was ~7% of single-thread
/// decode); stereo 16/24-bit, the common cases, get loops without per-sample
/// indexing that the compiler can vectorize.
pub fn append_interleaved<C: AsRef<[i64]>>(out: &mut Vec<u8>, channels: &[C], bits: u8) {
    let channels: Vec<&[i64]> = channels.iter().map(|c| c.as_ref()).collect();
    let nch = channels.len();
    let n = channels.first().map_or(0, |c| c.len());
    let bps = (bits / 8) as usize;
    let base = out.len();
    out.resize(base + n * nch * bps, 0);
    let dst = &mut out[base..];
    if nch == 0 || n == 0 { return; }
    let put = |d: &mut [u8], s: i64| match bits {
        8 => d[0] = (s + 128) as u8,
        16 => d.copy_from_slice(&(s as i16).to_le_bytes()),
        24 => d.copy_from_slice(&(s as i32).to_le_bytes()[..3]),
        _ => d.copy_from_slice(&(s as i32).to_le_bytes()),
    };
    match (nch, bits) {
        (2, 16) => for ((d, &l), &r) in dst.chunks_exact_mut(4).zip(&channels[0][..n]).zip(&channels[1][..n]) {
            d.copy_from_slice(&(((r as u16 as u32) << 16) | l as u16 as u32).to_le_bytes());
        },
        (2, 24) => for ((d, &l), &r) in dst.chunks_exact_mut(6).zip(&channels[0][..n]).zip(&channels[1][..n]) {
            let v = (l as u64 & 0xFF_FFFF) | ((r as u64 & 0xFF_FFFF) << 24);
            d.copy_from_slice(&v.to_le_bytes()[..6]);
        },
        (1, _) => for (d, &s) in dst.chunks_exact_mut(bps).zip(&channels[0][..n]) { put(d, s); },
        _ => for (i, frame) in dst.chunks_exact_mut(nch * bps).enumerate() {
            for (d, ch) in frame.chunks_exact_mut(bps).zip(&channels) { put(d, ch[i]); }
        },
    }
}

/// Appends raw float32 bit patterns ((b): already reconstructed by `floatpcm::unmap_from_pcm`,
/// never interpreted as a number here) as interleaved little-endian bytes.
pub fn append_interleaved_float(out: &mut Vec<u8>, channels: &[Vec<u32>]) {
    let nch = channels.len();
    let n = channels.first().map_or(0, |c| c.len());
    let base = out.len();
    out.resize(base + n * nch * 4, 0);
    let dst = &mut out[base..];
    if nch == 0 || n == 0 { return; }
    for (i, frame) in dst.chunks_exact_mut(nch * 4).enumerate() {
        for (d, ch) in frame.chunks_exact_mut(4).zip(channels) { d.copy_from_slice(&ch[i].to_le_bytes()); }
    }
}

pub fn write_wav<P: AsRef<Path>>(path: P, w: &Wav) -> Result<(), WavError> {
    let ch = w.channels.len() as u16;
    if ch == 0 { return Err(WavError("channels == 0".into())); }
    let n = w.channels[0].len();
    if w.channels.iter().any(|c| c.len() != n) { return Err(WavError("channel length mismatch".into())); }
    if let Some(info) = &w.float_info {
        let float_channels = floatpcm::unmap_from_pcm(&w.channels, info);
        let mut out = header_bytes_float(ch, w.sample_rate, w.channel_mask, n as u64)?;
        let data_start = out.len();
        append_interleaved_float(&mut out, &float_channels);
        if (out.len() - data_start) & 1 != 0 { out.push(0); }
        return fs::write(path, out).map_err(|e| WavError(e.to_string()));
    }
    let mut out = header_bytes(ch, w.sample_rate, w.bits, w.channel_mask, n as u64)?;
    let data_start = out.len();
    append_interleaved(&mut out, &w.channels, w.bits);
    if (out.len() - data_start) & 1 != 0 { out.push(0); }
    fs::write(path, out).map_err(|e| WavError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-channel strided form `append_interleaved` replaced: the reference.
    fn interleave_ref(channels: &[Vec<i64>], bits: u8) -> Vec<u8> {
        let (nch, n, bps) = (channels.len(), channels[0].len(), (bits / 8) as usize);
        let mut dst = vec![0u8; n * nch * bps];
        for (c, samples) in channels.iter().enumerate() {
            for (i, &s) in samples.iter().enumerate() {
                let o = (i * nch + c) * bps;
                match bits {
                    8 => dst[o] = (s + 128) as u8,
                    16 => dst[o..o + 2].copy_from_slice(&(s as i16).to_le_bytes()),
                    24 => dst[o..o + 3].copy_from_slice(&(s as i32).to_le_bytes()[..3]),
                    _ => dst[o..o + 4].copy_from_slice(&(s as i32).to_le_bytes()),
                }
            }
        }
        dst
    }

    #[test]
    fn interleave_matches_reference() {
        let mut st = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        for bits in [8u8, 16, 24, 32] {
            let (lo, hi) = (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1);
            for nch in 1..=6 {
                for n in [1usize, 2, 3, 17, 1000] {
                    let chans: Vec<Vec<i64>> = (0..nch).map(|_| (0..n).map(|i| match i % 5 {
                        0 => lo, 1 => hi, _ => lo + (next() % (1u64 << bits)) as i64,
                    }).collect()).collect();
                    let mut out = vec![0xAB];
                    append_interleaved(&mut out, &chans, bits);
                    assert_eq!(out[0], 0xAB);
                    assert_eq!(&out[1..], &interleave_ref(&chans, bits)[..], "bits {bits} nch {nch} n {n}");
                }
            }
        }
    }

    #[test]
    fn wav_roundtrip_all_bit_depths() {
        let dir = std::env::temp_dir();
        for bits in [8u8, 16, 24, 32] {
            let lo = -(1i64 << (bits - 1));
            let hi = (1i64 << (bits - 1)) - 1;
            let w = Wav { channels: vec![vec![lo, hi, 0, 1, -1, lo, hi], vec![hi, lo, 0, -1, 1, hi, lo]], sample_rate: 44100, bits, channel_mask: None, float_info: None };
            let p = dir.join(format!("nca_wavtest_{bits}.wav"));
            write_wav(&p, &w).unwrap();
            let back = read_wav(&p).unwrap();
            assert_eq!(back.channels, w.channels);
            assert_eq!(back.sample_rate, w.sample_rate);
            assert_eq!(back.bits, w.bits);
            assert_eq!(back.channel_mask, None);
            let _ = fs::remove_file(&p);
        }
    }

    #[test]
    fn channel_mask_roundtrips_through_wave_format_extensible() {
        let dir = std::env::temp_dir();
        // 5.1: front-left/front-right/front-center/LFE/back-left/back-right, the standard mask.
        const FIVE_POINT_ONE: u32 = 0x3F;
        let w = Wav {
            channels: (0..6).map(|c| vec![c as i64 * 100, -(c as i64) * 50]).collect(),
            sample_rate: 48000, bits: 24, channel_mask: Some(FIVE_POINT_ONE), float_info: None,
        };
        let p = dir.join("nca_wavtest_channel_mask.wav");
        write_wav(&p, &w).unwrap();
        let back = read_wav(&p).unwrap();
        assert_eq!(back.channels, w.channels);
        assert_eq!(back.channel_mask, Some(FIVE_POINT_ONE));
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn channel_mask_zero_is_distinct_from_absent() {
        // A real, legal WAVE_FORMAT_EXTENSIBLE file can declare mask=0 ("no defined layout"),
        // which must round-trip as Some(0), not collapse to None the way an Option easily could
        // if written carelessly (e.g. `if mask != 0`).
        let dir = std::env::temp_dir();
        let w = Wav { channels: vec![vec![1i64, 2, 3]], sample_rate: 44100, bits: 16, channel_mask: Some(0), float_info: None };
        let p = dir.join("nca_wavtest_channel_mask_zero.wav");
        write_wav(&p, &w).unwrap();
        let back = read_wav(&p).unwrap();
        assert_eq!(back.channel_mask, Some(0));
        let _ = fs::remove_file(&p);
    }
    /// The streaming reader across layouts real writers produce: extra chunks before
    /// and after fmt (odd-sized, so RIFF padding matters), the "unknown length" data sizes streamed
    /// WAVs use, a data size overrunning the file, a partial trailing frame, and data before fmt.
    #[test]
    fn streaming_reader_handles_real_world_layouts() {
        let w = Wav { channels: vec![vec![1i64, -2, 300, -32768, 32767], vec![5, 6, -7, 8, 9]], sample_rate: 44100, bits: 16, channel_mask: None, float_info: None };
        let mut body = header_bytes(2, 44100, 16, None, 5).unwrap();
        append_interleaved(&mut body, &w.channels, 16);
        let fmt_end = 12 + 8 + 16;
        let data_hdr = fmt_end;
        let odd = |id: &[u8; 4]| { let mut c = id.to_vec(); c.extend_from_slice(&3u32.to_le_bytes()); c.extend_from_slice(&[1, 2, 3, 0]); c };
        let parse = |b: &[u8]| read_wav_from(b, Some(b.len() as u64));
        // Chunks before fmt and between fmt and data.
        let mut v = body[..12].to_vec();
        v.extend(odd(b"LIST"));
        v.extend_from_slice(&body[12..fmt_end]);
        v.extend(odd(b"junk"));
        v.extend_from_slice(&body[data_hdr..]);
        assert_eq!(parse(&v).unwrap().channels, w.channels);
        // Streamed-WAV data sizes, and a size past the end of the file.
        for size in [0u32, 0xFFFF_FFFF, 1000] {
            let mut v = body.clone();
            v[data_hdr + 4..data_hdr + 8].copy_from_slice(&size.to_le_bytes());
            assert_eq!(parse(&v).unwrap().channels, w.channels, "size {size}");
            assert_eq!(read_wav_from(&v[..], None).unwrap().channels, w.channels, "size {size}, unknown length");
        }
        // A partial trailing frame is an error, as before.
        let mut v = body.clone();
        v.push(0);
        v[data_hdr + 4..data_hdr + 8].copy_from_slice(&0u32.to_le_bytes());
        assert!(parse(&v).is_err());
        // Data before fmt: rejected, not misread.
        let mut v = body[..12].to_vec();
        v.extend_from_slice(&body[data_hdr..]);
        v.extend_from_slice(&body[12..fmt_end]);
        assert!(parse(&v).is_err());
    }

    /// Reads that straddle the internal 1 MiB block boundary with a frame split across it.
    #[test]
    fn streaming_reader_joins_frames_across_blocks() {
        let n = 400_000usize;
        let w = Wav { channels: (0..3).map(|c| (0..n as i64).map(|i| ((i * 7919 + c * 13) % (1 << 23)) - (1 << 22)).collect()).collect(), sample_rate: 96000, bits: 24, channel_mask: Some(7), float_info: None };
        let mut b = header_bytes(3, 96000, 24, Some(7), n as u64).unwrap();
        append_interleaved(&mut b, &w.channels, 24);
        let back = read_wav_from(&b[..], Some(b.len() as u64)).unwrap();
        assert_eq!(back.channels, w.channels);
        assert_eq!(back.channel_mask, Some(7));
    }
}
