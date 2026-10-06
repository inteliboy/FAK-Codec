//! FAK container format: a fixed stream header, an optional metadata block, then a sequence of
//! self-delimiting chunks, each with its own inline sync, frame count, byte count and CRC
//! immediately before its payload. Every chunk decodes with no state from any other chunk, so encode
//! and decode both run one chunk per thread. `docs/bitstream-spec.md` is the specification.
use crate::crc::crc32;

pub const MAGIC: &[u8; 4] = b"FAK1";
/// The format version written to, and required in, every stream header (`docs/bitstream-spec.md`,
/// specification 1.1.0 after D-057 unfroze v20): a decoder accepts exactly this value.
pub const VERSION: u8 = 21;
/// Frame length code (format v15): `c` in 0..=6 means `FRAME_LEN_BASE << c` sample-frames;
/// `FRAME_LEN_ESCAPE` is followed by `FRAME_LEN_BITS` bits of `frame_frames - 1`.
pub const FRAME_LEN_BASE: u32 = 256;
pub const FRAME_LEN_ESCAPE: u64 = 7;
pub const FRAME_LEN_BITS: u32 = 20;

pub fn write_frame_len(w: &mut crate::bitio::BitWriter, n: u32) {
    debug_assert!(n > 0 && n <= MAX_FRAME_FRAMES);
    match (0..7).find(|&c| FRAME_LEN_BASE << c == n) {
        Some(c) => w.write_bits(c as u64, 3),
        None => { w.write_bits(FRAME_LEN_ESCAPE, 3); w.write_bits(n as u64 - 1, FRAME_LEN_BITS); }
    }
}

pub fn read_frame_len(r: &mut crate::bitio::BitReader) -> Result<u32, crate::bitio::BitReaderError> {
    let c = r.read_bits(3)?;
    Ok(if c == FRAME_LEN_ESCAPE { r.read_bits(FRAME_LEN_BITS)? as u32 + 1 } else { FRAME_LEN_BASE << c })
}
pub const DEFAULT_BLOCK_SIZE: usize = 4096;
/// Samples a block-mode subframe's predictor warmup may take from the preceding frames of the same
/// chunk (version 9): the longest predictor, `lpc::MAX_ORDER`.
pub const HISTORY_LEN: usize = crate::lpc::MAX_ORDER;
/// Sanity bound on a single frame's declared sample-frame count (defensive against a corrupted
/// or hostile length field triggering an oversized allocation).
pub const MAX_FRAME_FRAMES: u32 = 1 << 20;
pub const PCM_HASH_LEN: usize = 32;
// magic+version+channels+bits+mode+rate+total_frames+pcm_hash+crc
pub const HEADER_LEN: usize = 4 + 1 + 1 + 1 + 1 + 4 + 8 + PCM_HASH_LEN + 4;

/// Stream mode byte. Only `MODE_BLOCK_INDEPENDENT` exists: the forward-adaptive design (per-block
/// LPC/fixed predictors); each chunk is a run of self-delimiting frames. Mode 1, a backward-adaptive
/// cascade of adaptive filters (a), was removed in format v19
///; the byte stays in the header as a reserved field that must be 0, so a future stream mode
/// does not need a header change.
pub const MODE_BLOCK_INDEPENDENT: u8 = 0;

/// One chunk's descriptor: sample-frames in the chunk, payload byte length, CRC-32 of the payload.
/// Written inline immediately before the chunk's own payload (`write_chunk_header`), not batched
/// into an up-front table -- see the `VERSION` doc comment for why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkEntry {
    pub frames: u32,
    pub bytes: u32,
    pub crc: u32,
}

/// Sync word for one chunk's inline header -- lets a decoder confirm it has landed on a real chunk
/// boundary (not, say, a `bytes` field corrupted just enough to point mid-payload) before trusting
/// the frame/byte-length fields that follow, the same role the frame sync word played one level
/// down for frames within a `MODE_BLOCK_INDEPENDENT` chunk.
pub const CHUNK_SYNC: &[u8; 4] = b"FCHK";
/// sync(4) + frames(4) + bytes(4) + crc(4), immediately before each chunk's payload.
pub const CHUNK_HEADER_LEN: usize = 16;
/// Upper bound on one chunk's declared sample-frame count -- caps the per-chunk allocation a
/// hostile header can request. ~87s at 48 kHz, far above the encoder's default.
pub const MAX_CHUNK_FRAMES: u32 = 1 << 22;
/// Sentinel `total_frames` value meaning "unknown at encode time" -- a genuinely unbounded/live
/// source. A decoder reading such a stream keeps consuming self-delimited chunks until the
/// input itself is exhausted, rather than stopping at a pre-declared count.
pub const TOTAL_FRAMES_UNKNOWN: u64 = u64::MAX;
/// Target chunk duration the encoder uses by default: the trade-off between parallelism/seek
/// latency and per-chunk overhead. Not a format constraint -- the decoder accepts any chunking the
/// table describes.
///
/// Lowered 10 -> 1 (2026-09-26: smooth seeking while playing matters more than the compression
/// cost); block mode's chunking cost is only the ~12-byte-per-chunk table index.
pub const DEFAULT_CHUNK_SECONDS: usize = 1;

/// Default chunk length in sample-frames: `DEFAULT_CHUNK_SECONDS` rounded up to a whole number of
/// block-mode frames (so block-mode chunks contain only full-size frames except the last).
pub fn default_chunk_frames(sample_rate: u32) -> usize {
    let target = (sample_rate as usize).saturating_mul(DEFAULT_CHUNK_SECONDS).max(1);
    let blocks = target.div_ceil(DEFAULT_BLOCK_SIZE);
    (blocks * DEFAULT_BLOCK_SIZE).min(MAX_CHUNK_FRAMES as usize / DEFAULT_BLOCK_SIZE * DEFAULT_BLOCK_SIZE)
}

/// Serializes one chunk's inline header (`sync`, `frames`, `bytes`, `crc` -- all little-endian),
/// written immediately before that chunk's payload.
pub fn write_chunk_header(frames: u32, bytes: u32, crc: u32) -> [u8; CHUNK_HEADER_LEN] {
    let mut b = [0u8; CHUNK_HEADER_LEN];
    b[0..4].copy_from_slice(CHUNK_SYNC);
    b[4..8].copy_from_slice(&frames.to_le_bytes());
    b[8..12].copy_from_slice(&bytes.to_le_bytes());
    b[12..16].copy_from_slice(&crc.to_le_bytes());
    b
}

/// Parses one inline chunk header from the first `CHUNK_HEADER_LEN` bytes of `data`. Only checks
/// the sync word and decodes the fields -- bounds-checking `frames`/`bytes` against the rest of the
/// stream is the caller's job (`locate_chunks`, or `decoder::decode_stream`'s own equivalent
/// checks), since this function has no view of where the stream actually ends. `pub` (rather than
/// only used internally by `locate_chunks`) because the streaming decoder needs it too, reading one
/// fixed-size header buffer at a time from a `Read` source instead of slicing an in-memory buffer.
pub fn read_chunk_header(data: &[u8]) -> Result<ChunkEntry, FormatError> {
    if data.len() < CHUNK_HEADER_LEN { return Err(FormatError("truncated chunk header (corrupted stream?)".into())); }
    if &data[0..4] != CHUNK_SYNC { return Err(FormatError("bad chunk sync (corrupted stream?)".into())); }
    let frames = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let bytes = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let crc = u32::from_le_bytes(data[12..16].try_into().unwrap());
    Ok(ChunkEntry { frames, bytes, crc })
}

/// One located chunk: its first sample-frame's offset from the start of the stream, the byte
/// offset where its payload begins (right after its own inline header), and its descriptor.
pub type ChunkLoc = (u64, usize, ChunkEntry);

/// Sync word for one FEC parity block -- distinct from `CHUNK_SYNC` so `locate_chunks`'s walk (and
/// any other reader of the raw chunk sequence) can tell a parity block from a data chunk before
/// trusting either one's fields.
pub const PARITY_SYNC: &[u8; 4] = b"FPAR";
/// sync(4) + count(4) + m(2) + shard_len(4) + header crc(4), before the per-chunk table, the shard
/// CRCs and the shards.
pub const PARITY_FIXED_LEN: usize = 18;
/// One parity block's per-covered-chunk table entry: a redundant copy of that chunk's own
/// frames(4)+bytes(4)+crc(4) -- lets recovery learn a damaged chunk's real length and expected CRC
/// even though the damaged chunk's own copy of those fields might be what's unreliable.
pub const PARITY_ENTRY_LEN: usize = 12;
/// `fec_group` value meaning "one parity block for the whole file" (split only if the file has more
/// than [`MAX_FEC_GROUP`] chunks)..
pub const FEC_AUTO: usize = usize::MAX;
/// Default FEC layout: one Reed-Solomon parity block over the whole file, with
/// [`auto_parity`] shards. A smaller explicit group (`--fec-group N`) gives one block per N chunks.
pub const DEFAULT_FEC_GROUP: usize = FEC_AUTO;
/// A parity block covers at most this many chunks (the field holds `count + m <= 65536` points).
pub const MAX_FEC_GROUP: usize = 60000;
/// A parity block holds at most this many shards (recovering `e` chunks costs `O(e^3)` field ops).
pub const MAX_FEC_PARITY: usize = 1024;

static FEC_PARITY_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Process-wide encoder setting: parity shards per block (`None`/0 = [`auto_parity`]). Set by the
/// CLI's `--fec-parity`; the encode functions' signatures stay unchanged.
pub fn set_fec_parity(m: Option<usize>) { FEC_PARITY_OVERRIDE.store(m.unwrap_or(0), std::sync::atomic::Ordering::Relaxed); }

/// Parity shards for a block covering `count` chunks: the override if one is set, else about 1% of
/// the chunks (at least 2), never more than `count`. Any `m` chunks of the block can be lost and
/// rebuilt; the size cost is `m / count` of the compressed size.
pub fn auto_parity(count: usize) -> usize {
    let o = FEC_PARITY_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    let m = if o > 0 { o } else { count.div_ceil(100).max(2) };
    m.min(MAX_FEC_PARITY).min(count.max(1))
}

/// Chunks per parity block for a requested `fec_group`.
pub fn effective_group(fec_group: usize) -> usize { fec_group.clamp(1, MAX_FEC_GROUP) }

/// One located FEC parity block: which data-chunk indices (by position in `locate_chunks`'s own
/// returned `Vec<ChunkLoc>`) it covers, its redundant per-chunk table, and where its shards live.
/// Neither the header CRC nor the shard CRCs are verified here: that costs O(payload bytes) and a
/// healthy decode must not pay it -- recovery checks them
/// lazily, only after a covered chunk has already failed its own CRC.
#[derive(Clone, Debug)]
pub struct ParityLoc {
    pub first_chunk_idx: usize,
    pub entries: Vec<ChunkEntry>,
    pub m: usize,
    pub shard_len: usize,
    pub hdr_crc: u32,
    /// Byte offset of the table (entries, then the shard CRCs, then the shards).
    pub table_start: usize,
    pub shards_start: usize,
}

/// Computes a block's header CRC over its count, shard count, shard length, table and shard CRCs.
pub fn parity_hdr_crc(count: usize, m: usize, shard_len: usize, table_and_crcs: &[u8]) -> u32 {
    let mut b = Vec::with_capacity(10 + table_and_crcs.len());
    b.extend_from_slice(&(count as u32).to_le_bytes());
    b.extend_from_slice(&(m as u16).to_le_bytes());
    b.extend_from_slice(&(shard_len as u32).to_le_bytes());
    b.extend_from_slice(table_and_crcs);
    crc32(&b)
}

/// Builds one parity block as the group's chunks stream past: `m` Reed-Solomon shards
/// (`rs`) plus a redundant copy of every covered chunk's header fields.
pub struct ParityBuilder { acc: crate::rs::ShardAcc, entries: Vec<ChunkEntry> }

impl ParityBuilder {
    pub fn new(m: usize) -> Self { ParityBuilder { acc: crate::rs::ShardAcc::new(m), entries: Vec::new() } }
    pub fn count(&self) -> usize { self.entries.len() }
    pub fn push(&mut self, entry: ChunkEntry, payload: &[u8]) {
        self.entries.push(entry);
        self.acc.push(payload);
    }
    /// The finished block (an empty builder yields nothing). A short final group keeps only
    /// `min(m, count)` shards.
    pub fn finish(self) -> Result<Vec<u8>, FormatError> {
        let count = self.entries.len();
        if count == 0 { return Ok(Vec::new()); }
        let m = self.acc.m.min(count);
        if !crate::rs::fits(count, m) || count > MAX_FEC_GROUP || m > MAX_FEC_PARITY { return Err(FormatError("invalid FEC parity group".into())); }
        let shards: Vec<Vec<u8>> = self.acc.shards[..m].iter().map(|s| crate::rs::syms_to_bytes(s)).collect();
        let shard_len = shards[0].len();
        let mut tail = Vec::with_capacity(count * PARITY_ENTRY_LEN + m * 4);
        for e in &self.entries {
            tail.extend_from_slice(&e.frames.to_le_bytes());
            tail.extend_from_slice(&e.bytes.to_le_bytes());
            tail.extend_from_slice(&e.crc.to_le_bytes());
        }
        for s in &shards { tail.extend_from_slice(&crc32(s).to_le_bytes()); }
        let mut out = Vec::with_capacity(PARITY_FIXED_LEN + tail.len() + m * shard_len);
        out.extend_from_slice(PARITY_SYNC);
        out.extend_from_slice(&(count as u32).to_le_bytes());
        out.extend_from_slice(&(m as u16).to_le_bytes());
        out.extend_from_slice(&(shard_len as u32).to_le_bytes());
        out.extend_from_slice(&parity_hdr_crc(count, m, shard_len, &tail).to_le_bytes());
        out.extend_from_slice(&tail);
        for s in &shards { out.extend_from_slice(s); }
        Ok(out)
    }
}

/// A located-but-not-yet-content-verified parity block: `total_len` (bytes, for the caller to skip
/// past) and its framing.
pub struct ParityBlock { pub total_len: usize, pub entries: Vec<ChunkEntry>, pub m: usize, pub shard_len: usize, pub hdr_crc: u32, pub table_start: usize, pub shards_start: usize }

/// Parses one parity block's *framing* at `data[pos..]` -- an O(count) cost, the same class
/// `locate_chunks` already pays reading every data chunk's own 16-byte header.
pub fn read_parity_block(data: &[u8], pos: usize) -> Result<ParityBlock, FormatError> {
    let mut src = data;
    read_parity_block_src(&mut src, pos)
}

/// Random-access bytes of a stream: an in-memory slice, or a seekable file read on demand. The
/// container walk ([`locate_chunks_src`]) and FEC recovery are written once against this, so the
/// low-memory reader (`decoder::FileReader`) validates a hostile file exactly like the in-memory one.
pub trait ByteSource {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    /// Fills `buf` from byte `pos`; an error if that range is not entirely inside the stream.
    fn read_at(&mut self, pos: usize, buf: &mut [u8]) -> Result<(), FormatError>;
}

impl ByteSource for &[u8] {
    fn len(&self) -> usize { <[u8]>::len(self) }
    fn read_at(&mut self, pos: usize, buf: &mut [u8]) -> Result<(), FormatError> {
        let end = pos.checked_add(buf.len()).filter(|&e| e <= <[u8]>::len(self))
            .ok_or_else(|| FormatError("read past the end of the stream (corrupted stream?)".into()))?;
        buf.copy_from_slice(&self[pos..end]);
        Ok(())
    }
}

/// [`read_parity_block`] over any [`ByteSource`].
pub fn read_parity_block_src<S: ByteSource + ?Sized>(src: &mut S, pos: usize) -> Result<ParityBlock, FormatError> {
    let err = |m: &str| FormatError(format!("{m} (corrupted stream?)"));
    if pos.checked_add(PARITY_FIXED_LEN).is_none_or(|e| e > src.len()) { return Err(err("truncated FEC parity header")); }
    let mut fixed = [0u8; PARITY_FIXED_LEN];
    src.read_at(pos, &mut fixed)?;
    if &fixed[0..4] != PARITY_SYNC { return Err(err("bad FEC parity sync")); }
    let count = u32::from_le_bytes(fixed[4..8].try_into().unwrap()) as usize;
    let m = u16::from_le_bytes(fixed[8..10].try_into().unwrap()) as usize;
    let shard_len = u32::from_le_bytes(fixed[10..14].try_into().unwrap()) as usize;
    let hdr_crc = u32::from_le_bytes(fixed[14..18].try_into().unwrap());
    if count == 0 || count > MAX_FEC_GROUP { return Err(err("invalid FEC parity group size")); }
    if m == 0 || m > MAX_FEC_PARITY || m > count || !crate::rs::fits(count, m) { return Err(err("invalid FEC parity shard count")); }
    if shard_len % 2 != 0 { return Err(err("invalid FEC shard length")); }
    let table_start = pos + PARITY_FIXED_LEN;
    let table_len = count * PARITY_ENTRY_LEN;
    let shards_start = table_start.checked_add(table_len + m * 4).ok_or_else(|| err("FEC parity length overflow"))?;
    let end = m.checked_mul(shard_len).and_then(|t| shards_start.checked_add(t)).ok_or_else(|| err("FEC parity length overflow"))?;
    if end > src.len() { return Err(err("FEC parity block runs past end of file")); }
    let mut table = vec![0u8; table_len];
    src.read_at(table_start, &mut table)?;
    let mut entries = Vec::with_capacity(count);
    for c in table.chunks_exact(PARITY_ENTRY_LEN) {
        let frames = u32::from_le_bytes(c[0..4].try_into().unwrap());
        let bytes = u32::from_le_bytes(c[4..8].try_into().unwrap());
        let crc = u32::from_le_bytes(c[8..12].try_into().unwrap());
        entries.push(ChunkEntry { frames, bytes, crc });
    }
    Ok(ParityBlock { total_len: end - pos, entries, m, shard_len, hdr_crc, table_start, shards_start })
}

/// Walks the self-delimited chunk sequence starting at `data[payload_start..]`, validating every
/// field before trusting it: sync word, non-zero/bounded frame counts, in-bounds
/// payload lengths, and (for a known-length stream) that the frame counts sum exactly to
/// `header.total_frames` with no trailing garbage after the last chunk. For
/// `TOTAL_FRAMES_UNKNOWN` (a genuinely unbounded/streamed source) there is no total to
/// validate against -- the walk simply continues until the input itself is exhausted, and that is
/// the only valid stopping point. Never decodes any payload, only skips over it by byte offset, so
/// this is as cheap as parsing the old up-front table was -- O(chunk count), not O(stream length).
/// Since, a chunk boundary may also be an FEC parity block (`PARITY_SYNC`) rather than a data
/// chunk (`CHUNK_SYNC`) -- recognized and skipped the same cheap, O(count) way, with its own
/// `ParityLoc` recorded but its XOR-payload content left unverified (see `read_parity_block`).
/// Interior parity blocks (between two groups' worth of data chunks) are handled by the ordinary
/// walk below like any other boundary; a known-length stream's *final* group's parity block is a
/// special case, since `encoder::encode_chunked` writes it only after the last data chunk -- by
/// which point `frame_pos` has already reached `header.total_frames`, so it needs its own
/// "at most one trailing parity block, then nothing else" check rather than the loop's normal
/// per-chunk condition (which would otherwise treat reaching the declared total as the end of the
/// stream and misreport that trailing block as garbage).
///
/// Returns the located data chunks (in stream order), the located parity blocks (in stream order),
/// and the real total frame count found, which for a `TOTAL_FRAMES_UNKNOWN` stream is the only place
/// that total is ever known.
pub fn locate_chunks(header: &StreamHeader, data: &[u8], payload_start: usize) -> Result<(Vec<ChunkLoc>, Vec<ParityLoc>, u64), FormatError> {
    let mut src = data;
    locate_chunks_src(header, &mut src, payload_start)
}

/// [`locate_chunks`] over any [`ByteSource`]: reads only the 4-byte syncs, the 16-byte chunk headers
/// and the parity tables, and never a payload, so on a file it is O(chunk count) small reads.
pub fn locate_chunks_src<S: ByteSource + ?Sized>(header: &StreamHeader, src: &mut S, payload_start: usize) -> Result<(Vec<ChunkLoc>, Vec<ParityLoc>, u64), FormatError> {
    match walk_chunks_src(header, src, payload_start) {
        Ok(found) => Ok(found),
        // A damaged chunk header (sync, length) desyncs the walk; the parity blocks carry their own
        // copy of every chunk's table, so a file with FEC can still be located.
        Err(e) => locate_via_parity(header, src, payload_start).ok_or(e),
    }
}

/// Finds the chunks of a file whose inline chunk headers are unreliable, from its parity blocks:
/// scan for `FPAR` syncs, keep those whose header CRC (over the redundant chunk table and shard
/// CRCs) holds, and chain them -- each block must start exactly where the chunks its table lists
/// end. `None` unless the chain covers the whole file and the frame counts sum to `total_frames`.
fn locate_via_parity<S: ByteSource + ?Sized>(header: &StreamHeader, src: &mut S, payload_start: usize) -> Option<(Vec<ChunkLoc>, Vec<ParityLoc>, u64)> {
    const MAX_CANDIDATES: usize = 1024;
    let len = src.len();
    let mut blocks: Vec<(usize, ParityBlock)> = Vec::new();
    let mut tried = 0usize;
    let mut buf = vec![0u8; 1 << 20];
    let mut pos = payload_start;
    while pos + 4 <= len {
        let n = buf.len().min(len - pos);
        src.read_at(pos, &mut buf[..n]).ok()?;
        let mut i = 0;
        while let Some(off) = buf[i..n].windows(4).position(|w| w == PARITY_SYNC) {
            let at = pos + i + off;
            i += off + 1;
            tried += 1;
            if tried > MAX_CANDIDATES { return None; }
            let Ok(pb) = read_parity_block_src(src, at) else { continue };
            let tail_len = pb.entries.len() * PARITY_ENTRY_LEN + pb.m * 4;
            let mut tail = vec![0u8; tail_len];
            if src.read_at(pb.table_start, &mut tail).is_err() { continue; }
            if parity_hdr_crc(pb.entries.len(), pb.m, pb.shard_len, &tail) != pb.hdr_crc { continue; }
            blocks.push((at, pb));
        }
        if pos + n >= len { break; }
        pos += n - 3; // a sync may straddle the buffer boundary
    }
    blocks.sort_by_key(|b| b.0);
    blocks.dedup_by_key(|b| b.0);
    let mut chunks: Vec<ChunkLoc> = Vec::new();
    let mut parities = Vec::new();
    let mut cur = payload_start;
    let mut frame_pos = 0u64;
    for (at, pb) in blocks {
        let first_chunk_idx = chunks.len();
        let mut p = cur;
        for e in &pb.entries {
            if e.frames == 0 || e.frames > MAX_CHUNK_FRAMES { return None; }
            chunks.push((frame_pos, p + CHUNK_HEADER_LEN, *e));
            frame_pos = frame_pos.checked_add(e.frames as u64)?;
            p = p.checked_add(CHUNK_HEADER_LEN)?.checked_add(e.bytes as usize)?;
        }
        if p != at { return None; }
        cur = at + pb.total_len;
        parities.push(ParityLoc { first_chunk_idx, entries: pb.entries, m: pb.m, shard_len: pb.shard_len, hdr_crc: pb.hdr_crc, table_start: pb.table_start, shards_start: pb.shards_start });
    }
    if cur != len || chunks.is_empty() { return None; }
    if header.total_frames != TOTAL_FRAMES_UNKNOWN && frame_pos != header.total_frames { return None; }
    Some((chunks, parities, frame_pos))
}

fn walk_chunks_src<S: ByteSource + ?Sized>(header: &StreamHeader, src: &mut S, payload_start: usize) -> Result<(Vec<ChunkLoc>, Vec<ParityLoc>, u64), FormatError> {
    let err = |m: &str| FormatError(format!("{m} (corrupted stream?)"));
    let len = src.len();
    let mut out = Vec::new();
    let mut parities = Vec::new();
    let mut pos = payload_start;
    let mut frame_pos = 0u64;
    let mut trailing_parity_done = false;
    while pos < len {
        let known_done = header.total_frames != TOTAL_FRAMES_UNKNOWN && frame_pos >= header.total_frames;
        let mut sync = [0u8; 4];
        if known_done {
            if trailing_parity_done || pos + 4 > len { break; }
            src.read_at(pos, &mut sync)?;
            if &sync != PARITY_SYNC { break; }
            let pb = read_parity_block_src(src, pos)?;
            let first_chunk_idx = out.len().checked_sub(pb.entries.len())
                .ok_or_else(|| err("FEC parity group size exceeds preceding chunk count"))?;
            parities.push(ParityLoc { first_chunk_idx, entries: pb.entries, m: pb.m, shard_len: pb.shard_len, hdr_crc: pb.hdr_crc, table_start: pb.table_start, shards_start: pb.shards_start });
            pos += pb.total_len;
            trailing_parity_done = true;
            continue;
        }
        if pos + 4 > len { return Err(err("truncated chunk/parity sync")); }
        src.read_at(pos, &mut sync)?;
        if &sync == CHUNK_SYNC {
            if pos + CHUNK_HEADER_LEN > len { return Err(err("truncated chunk header")); }
            let mut hb = [0u8; CHUNK_HEADER_LEN];
            hb[..4].copy_from_slice(&sync);
            src.read_at(pos + 4, &mut hb[4..])?;
            let e = read_chunk_header(&hb)?;
            if e.frames == 0 || e.frames > MAX_CHUNK_FRAMES { return Err(err(&format!("invalid chunk frame count {}", e.frames))); }
            let byte_start = pos + CHUNK_HEADER_LEN;
            let payload_end = byte_start.checked_add(e.bytes as usize).ok_or_else(|| err("chunk length overflow"))?;
            if payload_end > len { return Err(err("chunk payload runs past end of file")); }
            out.push((frame_pos, byte_start, e));
            frame_pos = frame_pos.checked_add(e.frames as u64).ok_or_else(|| err("frame count overflow"))?;
            pos = payload_end;
        } else if &sync == PARITY_SYNC {
            let pb = read_parity_block_src(src, pos)?;
            let first_chunk_idx = out.len().checked_sub(pb.entries.len())
                .ok_or_else(|| err("FEC parity group size exceeds preceding chunk count"))?;
            parities.push(ParityLoc { first_chunk_idx, entries: pb.entries, m: pb.m, shard_len: pb.shard_len, hdr_crc: pb.hdr_crc, table_start: pb.table_start, shards_start: pb.shards_start });
            pos += pb.total_len;
        } else {
            return Err(err("bad chunk/parity sync"));
        }
        if header.total_frames != TOTAL_FRAMES_UNKNOWN && frame_pos > header.total_frames {
            return Err(err("chunk frame counts exceed total_frames"));
        }
    }
    if header.total_frames != TOTAL_FRAMES_UNKNOWN {
        if frame_pos != header.total_frames { return Err(err("chunk frame counts don't sum to total_frames")); }
        if pos != len { return Err(err("trailing garbage after chunk data")); }
    }
    Ok((out, parities, frame_pos))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubframeType { Constant = 0, Verbatim = 1, Fixed = 2, Lpc = 3, Palette = 4, PaletteRle = 5, Cross = 6 }

impl SubframeType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Constant), 1 => Some(Self::Verbatim), 2 => Some(Self::Fixed),
            3 => Some(Self::Lpc), 4 => Some(Self::Palette), 5 => Some(Self::PaletteRle), 6 => Some(Self::Cross), _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamHeader {
    pub channels: u8,
    pub bits_per_sample: u8,
    pub mode: u8,
    pub sample_rate: u32,
    pub total_frames: u64,
    /// SHA-256 of the decoded PCM (`sha256::pcm_digest`),. Detection-only, checked by
    /// `decoder::verify`, never automatically -- see the `VERSION` doc comment above.
    pub pcm_hash: [u8; PCM_HASH_LEN],
}

#[derive(Debug)]
pub struct FormatError(pub String);
impl std::fmt::Display for FormatError { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "{}", self.0) } }
impl std::error::Error for FormatError {}

impl StreamHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(HEADER_LEN);
        b.extend_from_slice(MAGIC);
        b.push(VERSION);
        b.push(self.channels);
        b.push(self.bits_per_sample);
        b.push(self.mode);
        b.extend_from_slice(&self.sample_rate.to_le_bytes());
        b.extend_from_slice(&self.total_frames.to_le_bytes());
        b.extend_from_slice(&self.pcm_hash);
        let crc = crc32(&b[4..]);
        b.extend_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, FormatError> {
        if data.len() < HEADER_LEN { return Err(FormatError("truncated header".into())); }
        if &data[0..4] != MAGIC { return Err(FormatError("bad magic".into())); }
        let version = data[4];
        if version != VERSION { return Err(FormatError(format!("unsupported version {version}"))); }
        let channels = data[5];
        let bits_per_sample = data[6];
        let mode = data[7];
        if channels == 0 { return Err(FormatError("channels == 0".into())); }
        if bits_per_sample != 8 && bits_per_sample != 16 && bits_per_sample != 24 && bits_per_sample != 32 {
            return Err(FormatError(format!("unsupported bits_per_sample {bits_per_sample}")));
        }
        if mode != MODE_BLOCK_INDEPENDENT {
            return Err(FormatError(format!("unsupported stream mode {mode}")));
        }
        let sample_rate = u32::from_le_bytes(data[8..12].try_into().unwrap());
        if sample_rate == 0 { return Err(FormatError("sample_rate == 0".into())); }
        let total_frames = u64::from_le_bytes(data[12..20].try_into().unwrap());
        let pcm_hash: [u8; PCM_HASH_LEN] = data[20..20 + PCM_HASH_LEN].try_into().unwrap();
        let crc_at = 20 + PCM_HASH_LEN;
        let stored_crc = u32::from_le_bytes(data[crc_at..crc_at + 4].try_into().unwrap());
        let computed = crc32(&data[4..crc_at]);
        if computed != stored_crc { return Err(FormatError("header CRC mismatch (corrupted file?)".into())); }
        Ok(StreamHeader { channels, bits_per_sample, mode, sample_rate, total_frames, pcm_hash })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let h = StreamHeader { channels: 2, bits_per_sample: 16, mode: MODE_BLOCK_INDEPENDENT, sample_rate: 44100, total_frames: 123_456, pcm_hash: [7u8; PCM_HASH_LEN] };
        let b = h.to_bytes();
        assert_eq!(b.len(), HEADER_LEN);
        let h2 = StreamHeader::from_bytes(&b).unwrap();
        assert_eq!(h, h2);
    }

    #[test]
    fn header_roundtrip_32bit_block_mode() {
        let h = StreamHeader { channels: 2, bits_per_sample: 32, mode: MODE_BLOCK_INDEPENDENT, sample_rate: 48000, total_frames: 1000, pcm_hash: [3u8; PCM_HASH_LEN] };
        let b = h.to_bytes();
        let h2 = StreamHeader::from_bytes(&b).unwrap();
        assert_eq!(h, h2);
    }

    /// Mode 1 (the removed backward-adaptive mode) and any other non-zero mode byte is refused.
    #[test]
    fn header_rejects_the_removed_backward_mode() {
        let h = StreamHeader { channels: 2, bits_per_sample: 16, mode: 1, sample_rate: 48000, total_frames: 1000, pcm_hash: [3u8; PCM_HASH_LEN] };
        assert!(StreamHeader::from_bytes(&h.to_bytes()).is_err());
    }

    #[test]
    fn header_rejects_unknown_mode() {
        let h = StreamHeader { channels: 1, bits_per_sample: 16, mode: 2, sample_rate: 44100, total_frames: 0, pcm_hash: [0u8; PCM_HASH_LEN] };
        assert!(StreamHeader::from_bytes(&h.to_bytes()).is_err());
    }

    #[test]
    fn header_rejects_bad_magic_and_corruption() {
        let h = StreamHeader { channels: 1, bits_per_sample: 24, mode: MODE_BLOCK_INDEPENDENT, sample_rate: 96000, total_frames: 0, pcm_hash: [0u8; PCM_HASH_LEN] };
        let mut b = h.to_bytes();
        assert!(StreamHeader::from_bytes(&b[..4]).is_err()); // truncated
        b[0] = b'X';
        assert!(StreamHeader::from_bytes(&b).is_err()); // bad magic
        let mut b2 = h.to_bytes();
        b2[9] ^= 0xFF; // flip a byte inside sample_rate, after magic/version so CRC must catch it
        assert!(StreamHeader::from_bytes(&b2).is_err());
    }

    #[test]
    fn header_rejects_invalid_fields() {
        let mut h = StreamHeader { channels: 0, bits_per_sample: 16, mode: MODE_BLOCK_INDEPENDENT, sample_rate: 44100, total_frames: 0, pcm_hash: [0u8; PCM_HASH_LEN] };
        assert!(StreamHeader::from_bytes(&h.to_bytes()).is_err());
        h.channels = 2;
        h.bits_per_sample = 20; // not 8/16/24
        assert!(StreamHeader::from_bytes(&h.to_bytes()).is_err());
    }

    #[test]
    fn header_crc_covers_pcm_hash() {
        let h = StreamHeader { channels: 2, bits_per_sample: 16, mode: MODE_BLOCK_INDEPENDENT, sample_rate: 44100, total_frames: 10, pcm_hash: [1u8; PCM_HASH_LEN] };
        let mut b = h.to_bytes();
        b[20] ^= 0xFF; // flip a byte inside pcm_hash
        assert!(StreamHeader::from_bytes(&b).is_err(), "corrupting pcm_hash must be caught by the header CRC");
    }
}
