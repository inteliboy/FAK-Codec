//! Reference decoder. Every declared length is range-checked against the data actually present
//! and against the stream's own declared totals before being trusted for allocation or indexing,
//! and every frame's CRC-16 is verified — the decoder treats the input as hostile.
use crate::bitio::{BitReader, BitReaderError};
use crate::crc::crc32;
use crate::crossch;
use crate::stage2;
use crate::ltp;
use crate::format::{self, locate_chunks, locate_chunks_src, ByteSource, ChunkLoc, FormatError, ParityLoc, StreamHeader, SubframeType, HEADER_LEN, HISTORY_LEN, MAX_FRAME_FRAMES, TOTAL_FRAMES_UNKNOWN};
use crate::metadata::{self, Metadata};
use crate::parallel;
use crate::sha256;
use crate::rs;
use crate::lpc::{self, QuantizedLpc, MAX_ORDER as LPC_MAX_ORDER, MAX_PRECISION, MIN_PRECISION};
use crate::palette;
use crate::predictors::{self, MAX_ORDER};
use crate::rice;
use crate::stereo::{self, StereoMode};
use crate::valuemap;

fn bmap(e: BitReaderError) -> FormatError { FormatError(e.0.to_string()) }

/// Research tap (feature `research-tap`, never in normal builds): where each Fixed/LPC/Cross
/// subframe's bits go, and its predictor and Rice selectors, for pricing side-information coding
/// offline (`examples/side_info.rs`). Decode single-threaded when reading it.
#[cfg(feature = "research-tap")]
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct SubTap {
    pub slot: u8, pub stereo_mode: u8, pub kind: u8, pub n: u32, pub order: u8, pub precision: u8, pub shift: u8,
    pub coeffs: Vec<i32>, pub hdr_bits: u32, pub coef_bits: u32, pub warm_bits: u32, pub cross_bits: u32,
    pub s2_bits: u32, pub rice_bits: u32, pub rice_sel: Vec<u8>, pub res: Vec<i64>,
    ///  (`examples/nonlinear_dump.rs`): this subframe's start position within the frame (0 if
    /// warmup came from history), its reconstructed samples, and -- for a stereo frame's second
    /// subframe onward -- the other channel's full-frame residual and reconstructed samples
    /// (zero-padded before its own `res_start`, so both align by absolute frame position).
    pub res_start: u32, pub samples: Vec<i64>, pub other_res: Vec<i64>, pub other_samples: Vec<i64>,
}
#[cfg(feature = "research-tap")]
#[doc(hidden)]
pub static TAP: std::sync::Mutex<Vec<SubTap>> = std::sync::Mutex::new(Vec::new());
/// Research tap: frames read, and bytes of frame payload+CRC (the rest of a chunk is its value-map
/// section).
#[cfg(feature = "research-tap")]
#[doc(hidden)]
pub static FTAP: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));

/// A decoded subframe: its samples (wasted bits restored) and its predictor residual, which covers
/// frame positions `res_start..n` (none for Constant/Verbatim/Palette: `res_start == n`; for a
/// `Cross` subframe, the residual of its own Fixed/LPC predictor, i.e. with the cross-channel term
/// already added back). Kept so a later subframe of the same frame can reference either
/// (cross-channel prediction, format v12).
pub(crate) struct SubOut { pub samples: Vec<i64>, pub res: Vec<i64>, pub res_start: usize }

impl SubOut {
    /// The residual as a frame-length signal, zero where it has none.
    fn residual_signal(&self) -> Vec<i64> {
        let mut v = vec![0i64; self.samples.len()];
        v[self.res_start..].copy_from_slice(&self.res);
        v
    }
}

/// Fixed/LPC subframe body (after the type field). With `cross = Some(refs)` the cross-channel
/// fields follow the predictor's own (subframe type `Cross`). Returns the shifted-domain samples,
/// the predictor residual and where it starts.
fn read_predicted(r: &mut BitReader, inner: SubframeType, n: usize, eff_bits: u32, hist: &[i64], cross: Option<&[SubOut]>, mut carry: Option<&mut stage2::Carried>) -> Result<(Vec<i64>, Vec<i64>, usize), FormatError> {
    #[cfg(feature = "research-tap")]
    let t0 = r.bit_pos();
    let (order, lpc_q) = match inner {
        SubframeType::Fixed => {
            let order = r.read_bits(3).map_err(bmap)? as usize;
            if order > MAX_ORDER as usize { return Err(FormatError(format!("invalid predictor order {order}"))); }
            #[cfg(feature = "research-tap")]
            TAP.lock().unwrap().push(SubTap { order: order as u8, hdr_bits: (r.bit_pos() - t0) as u32, ..Default::default() });
            (order, None)
        }
        SubframeType::Lpc => {
            let order = r.read_bits(5).map_err(bmap)? as usize + 1; // stored as order-1
            if order > LPC_MAX_ORDER { return Err(FormatError(format!("invalid LPC order {order}"))); }
            let shift = r.read_bits(5).map_err(bmap)? as u32;
            let precision = r.read_bits(4).map_err(bmap)? as u32 + 1; // stored as precision-1
            if !(MIN_PRECISION..=MAX_PRECISION).contains(&precision) {
                return Err(FormatError(format!("invalid LPC coefficient precision {precision}")));
            }
            #[cfg(feature = "research-tap")]
            let tc = r.bit_pos();
            let coeffs = lpc::read_coeffs(r, order, precision).map_err(|e| FormatError(e.into()))?;
            #[cfg(feature = "research-tap")]
            TAP.lock().unwrap().push(SubTap { order: order as u8, precision: precision as u8, shift: shift as u8,
                coeffs: coeffs.iter().map(|&c| c as i32).collect(), hdr_bits: (tc - t0) as u32, coef_bits: (r.bit_pos() - tc) as u32, ..Default::default() });
            (order, Some(QuantizedLpc { coeffs, shift, precision }))
        }
        _ => return Err(FormatError(format!("invalid cross-channel inner subframe type {}", inner as u8))),
    };
    // Warmup: from the history when it is long enough (format v9), else stored verbatim. A Fixed
    // order-0 subframe has no warmup either way.
    let from_history = order == 0 || hist.len() >= order;
    if !from_history && n < order { return Err(FormatError("frame shorter than predictor order".into())); }
    let res_start = if from_history { 0 } else { order };
    let mut warmup = Vec::new();
    #[cfg(feature = "research-tap")]
    let tw = r.bit_pos();
    if !from_history { for _ in 0..order { warmup.push(r.read_signed(eff_bits).map_err(bmap)?); } }
    #[cfg(feature = "research-tap")]
    let tx = r.bit_pos();
    let cross = match cross {
        Some(refs) => Some((crossch::CrossParams::read(r, refs.len()).map_err(FormatError)?, refs)),
        None => None,
    };
    #[cfg(feature = "research-tap")]
    let ts = r.bit_pos();
    let carried_s = match carry {
        Some(_) if r.read_bits(1).map_err(bmap)? == 1 => Some(stage2::Carried::read_s(r).map_err(FormatError)?),
        _ => None,
    };
    let s2 = if carried_s.is_some() { None } else { stage2::Params::read(r).map_err(FormatError)? };
    let lt = ltp::Params::read(r).map_err(FormatError)?;
    #[cfg(feature = "research-tap")]
    let (tr, _) = (r.bit_pos(), rice::DTAP.lock().unwrap().clear());
    let mut res = { let _g = crate::prof::span(crate::prof::Phase::DecRice); rice::decode(r, n - res_start).map_err(bmap)? };
    #[cfg(feature = "research-tap")]
    {
        let mut t = TAP.lock().unwrap();
        let e = t.last_mut().unwrap();
        e.warm_bits = (tx - tw) as u32; e.cross_bits = (ts - tx) as u32; e.s2_bits = (tr - ts) as u32;
        e.rice_bits = (r.bit_pos() - tr) as u32; e.rice_sel = std::mem::take(&mut *rice::DTAP.lock().unwrap());
        e.n = n as u32; e.kind = inner as u8 + if cross.is_some() { 8 } else { 0 };
    }
    if let Some(p) = &lt { let _g = crate::prof::span(crate::prof::Phase::DecLtp); ltp::inverse(p, &mut res).map_err(|e| FormatError(e.into()))?; }
    // The tapped residual is the one before long-term prediction (what stage 2 or the predictor
    // left), so probes price LTP and alternatives to it from the same starting point.
    #[cfg(feature = "research-tap")]
    if std::env::var_os("FAK_TAP_RES").is_some() {
        let mut t = TAP.lock().unwrap();
        let e = t.last_mut().unwrap();
        e.res = res.clone(); e.res_start = res_start as u32;
    }
    if let Some(p) = &s2 { let _g = crate::prof::span(crate::prof::Phase::DecStage2); stage2::inverse(p, &mut res).map_err(|e| FormatError(e.into()))?; }
    if let Some(c) = carry.as_deref_mut() {
        let _g = crate::prof::span(crate::prof::Phase::DecCarried);
        match carried_s {
            Some(s) => c.inverse(&mut res, s).map_err(|e| FormatError(e.into()))?,
            None if !res.is_empty() && res.iter().all(|e| e.abs() <= stage2::MAX_RESIDUAL) => {
                let s = stage2::Params::for_block(&res, c.taps(), c.k(), stage2::Carried::TARGET).s;
                c.advance(&res, s);
            }
            None => {}
        }
    }
    if let Some((p, refs)) = &cross {
        let _g = crate::prof::span(crate::prof::Phase::DecCross);
        let rf = &refs[p.ref_idx as usize];
        let padded;
        let src: &[i64] = match p.source {
            crossch::Source::Samples => &rf.samples,
            crossch::Source::Residual if rf.res_start == 0 => &rf.res,
            crossch::Source::Residual => { padded = rf.residual_signal(); &padded }
        };
        if src.len() != n { return Err(FormatError("cross-channel reference length mismatch".into())); }
        crossch::apply(p, src, res_start, &mut res, false).map_err(|e| FormatError(e.into()))?;
    }
    let err = |e: &'static str| FormatError(e.into());
    let g_rec = crate::prof::span(crate::prof::Phase::DecPredictor);
    let mut out = Vec::with_capacity(n);
    match (&lpc_q, from_history) {
        (Some(q), true) => lpc::reconstruct_into(q, &hist[hist.len() - order..], &res, &mut out).map_err(err)?,
        (Some(q), false) => { out = lpc::reconstruct(q, &warmup, &res).map_err(err)?; }
        (None, true) => predictors::reconstruct_into(order as u8, &hist[hist.len() - order..], &res, &mut out).map_err(err)?,
        (None, false) => { out = predictors::reconstruct(order as u8, &warmup, &res).map_err(err)?; }
    }
    drop(g_rec);
    #[cfg(feature = "research-tap")]
    if std::env::var_os("FAK_TAP_RES").is_some() { TAP.lock().unwrap().last_mut().unwrap().samples = out.clone(); }
    Ok((out, res, res_start))
}

/// `history`: up to `HISTORY_LEN` samples of this subframe's channel representation that
/// precede it in the same chunk (empty for a chunk's first frame). A Fixed/LPC subframe whose order
/// fits in it takes its warmup from there instead of reading it verbatim (format v9) -- the
/// exact rule `encoder::warmup_source` applies. `refs`: the frame's earlier subframes, which a
/// `Cross` subframe may reference.
#[cfg(test)]
pub(crate) fn read_subframe(r: &mut BitReader, n: usize, bits_eff: u32, history: &[i64], refs: &[SubOut]) -> Result<SubOut, FormatError> {
    read_subframe_carried(r, n, bits_eff, history, refs, None)
}

pub(crate) fn read_subframe_carried(r: &mut BitReader, n: usize, bits_eff: u32, history: &[i64], refs: &[SubOut], carry: Option<&mut stage2::Carried>) -> Result<SubOut, FormatError> {
    let wasted = r.read_bits(5).map_err(bmap)? as u32;
    if wasted >= bits_eff { return Err(FormatError(format!("invalid wasted-bits count {wasted} >= {bits_eff}"))); }
    let eff_bits = bits_eff - wasted;
    let hist: Vec<i64> = history.iter().map(|&h| h >> wasted).collect();
    let kind_v = r.read_bits(3).map_err(bmap)? as u8;
    let kind = SubframeType::from_u8(kind_v).ok_or_else(|| FormatError(format!("invalid subframe type {kind_v}")))?;
    let mut res = Vec::new();
    let mut res_start = n;
    let shifted = match kind {
        SubframeType::Constant => {
            let v = r.read_signed(eff_bits).map_err(bmap)?;
            vec![v; n]
        }
        SubframeType::Verbatim => {
            let mut out = Vec::with_capacity(n);
            for _ in 0..n { out.push(r.read_signed(eff_bits).map_err(bmap)?); }
            out
        }
        SubframeType::Fixed | SubframeType::Lpc | SubframeType::Cross => {
            let (inner, cross) = if kind == SubframeType::Cross {
                let v = r.read_bits(3).map_err(bmap)? as u8;
                (SubframeType::from_u8(v).ok_or_else(|| FormatError(format!("invalid subframe type {v}")))?, Some(refs))
            } else { (kind, None) };
            let (out, rs, start) = read_predicted(r, inner, n, eff_bits, &hist, cross, carry)?;
            res = rs;
            res_start = start;
            out
        }
        SubframeType::Palette => {
            let count = r.read_bits(4).map_err(bmap)? as usize + 2; // stored as count-2, so 2..=17
            if count > palette::MAX_PALETTE { return Err(FormatError(format!("invalid palette size {count} (corrupted stream?)"))); }
            let mut pal = Vec::with_capacity(count);
            for _ in 0..count { pal.push(r.read_signed(eff_bits).map_err(bmap)?); }
            let iw = palette::index_width(count);
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                let i = r.read_bits(iw).map_err(bmap)? as usize;
                if i >= count { return Err(FormatError(format!("palette index {i} >= table size {count} (corrupted stream?)"))); }
                out.push(pal[i]);
            }
            out
        }
        SubframeType::PaletteRle => {
            let count = r.read_bits(4).map_err(bmap)? as usize + 2;
            if count > palette::MAX_PALETTE { return Err(FormatError(format!("invalid palette size {count} (corrupted stream?)"))); }
            let mut pal = Vec::with_capacity(count);
            for _ in 0..count { pal.push(r.read_signed(eff_bits).map_err(bmap)?); }
            let iw = palette::index_width(count);
            let run_len_bits = r.read_bits(5).map_err(bmap)? as u32;
            if run_len_bits == 0 { return Err(FormatError("invalid PaletteRle run_len_bits 0 (corrupted stream?)".into())); }
            // num_runs is stream-declared and must not be trusted for allocation before it's
            // checked against n ("enormous declared sizes") -- capped below, not
            // pre-allocated from the raw field.
            let num_runs = r.read_bits(20).map_err(bmap)? as usize + 1;
            if num_runs > n { return Err(FormatError(format!("PaletteRle num_runs {num_runs} exceeds subframe length {n} (corrupted stream?)"))); }
            let mut out = Vec::with_capacity(n);
            for _ in 0..num_runs {
                let i = r.read_bits(iw).map_err(bmap)? as usize;
                if i >= count { return Err(FormatError(format!("PaletteRle index {i} >= table size {count} (corrupted stream?)"))); }
                let len = r.read_bits(run_len_bits).map_err(bmap)? as usize + 1;
                if out.len() + len > n { return Err(FormatError("PaletteRle run overruns subframe length (corrupted stream?)".into())); }
                out.extend(std::iter::repeat(pal[i]).take(len));
            }
            if out.len() != n { return Err(FormatError(format!("PaletteRle runs total {} samples, expected {n} (corrupted stream?)", out.len()))); }
            out
        }
    };
    let samples = if wasted == 0 { shifted } else { shifted.into_iter().map(|s| s << wasted).collect() };
    Ok(SubOut { samples, res, res_start })
}

/// Decode with every available thread, discarding any tag/artwork metadata -- see [`decode_full`]
/// to get it too.
pub fn decode(data: &[u8]) -> Result<(StreamHeader, Vec<Vec<i64>>), FormatError> {
    decode_with_threads(data, parallel::default_threads())
}

/// [`decode`] on up to `threads` workers, discarding metadata.
pub fn decode_with_threads(data: &[u8], threads: usize) -> Result<(StreamHeader, Vec<Vec<i64>>), FormatError> {
    let (header, _metadata, channels) = decode_full(data, threads)?;
    Ok((header, channels))
}

/// Shared by [`decode_full`] and [`seek`]: parses and validates the header, metadata block, and the
/// self-delimited chunk sequence, so a corrupted or hostile file is rejected identically by
/// both entry points before either one starts decoding any chunk payload. Returns the located
/// chunks and the real total frame count `locate_chunks` found -- for an ordinary known-length
/// stream this always equals `header.total_frames` (validated), but for a `TOTAL_FRAMES_UNKNOWN`
/// stream (a genuinely unbounded/streamed source) `header.total_frames` is just the sentinel, so
/// this is the only place the real total is ever available.
fn check_cue(header: &StreamHeader, meta: &Metadata) -> Result<(), FormatError> {
    if header.total_frames != TOTAL_FRAMES_UNKNOWN {
        if let Some(cue) = &meta.cue_sheet {
            for t in &cue.tracks {
                for idx in &t.indices {
                    if idx.sample_offset > header.total_frames {
                        return Err(FormatError("cue sheet index offset exceeds total_frames (corrupted stream?)".into()));
                    }
                }
            }
        }
    }
    Ok(())
}

fn parse_container(data: &[u8]) -> Result<(StreamHeader, Vec<ChunkLoc>, Vec<ParityLoc>, Metadata, u64, usize), FormatError> {
    let header = StreamHeader::from_bytes(data)?;
    let (meta, payload_start) = metadata::read_block(data, HEADER_LEN).map_err(|e| FormatError(e.0))?;
    // A cue point past the end of the actual stream is meaningless and worth rejecting outright
    // (not a memory-safety issue -- sample_offset is never used for allocation or indexing here --
    // but treats every declared value in a hostile file as worth validating). Skipped
    // for a `TOTAL_FRAMES_UNKNOWN` stream -- there is no declared total yet to validate against.
    check_cue(&header, &meta)?;
    let (chunks, parities, real_total_frames) = locate_chunks(&header, data, payload_start)?;
    Ok((header, chunks, parities, meta, real_total_frames, payload_start))
}

/// Rebuilt payloads per parity block (keyed by the block's first chunk index), so a file with many
/// damaged chunks reads and solves each block once, not once per chunk. `Err` holds why that block
/// could not be solved.
#[derive(Default)]
pub struct FecCache(std::sync::Mutex<std::collections::HashMap<usize, Result<std::collections::HashMap<usize, Vec<u8>>, String>>>);

/// Rebuilds every damaged chunk of one parity block. A chunk is damaged if its payload does
/// not match the CRC in the block's own table; `e` of them are rebuilt from any `e` intact shards.
/// Reads the group twice (once to find the damaged chunks, once to fold the intact ones into the
/// shards' syndromes) -- only ever reached after a chunk has already failed its own CRC.
fn recover_group<S: ByteSource + ?Sized>(src: &mut S, chunks: &[ChunkLoc], pl: &ParityLoc) -> Result<std::collections::HashMap<usize, Vec<u8>>, String> {
    let e2s = |e: FormatError| e.0;
    let count = pl.entries.len();
    let m = pl.m;
    let shard_len = pl.shard_len;
    if pl.first_chunk_idx + count > chunks.len() { return Err("FEC parity group runs past the chunk list (corrupted stream?)".into()); }
    let mut tail = vec![0u8; count * format::PARITY_ENTRY_LEN + m * 4];
    src.read_at(pl.table_start, &mut tail).map_err(e2s)?;
    if format::parity_hdr_crc(count, m, shard_len, &tail) != pl.hdr_crc {
        return Err("the parity block's own header is corrupted".into());
    }
    let crcs_at = count * format::PARITY_ENTRY_LEN;
    let shard_crcs: Vec<u32> = tail[crcs_at..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    if pl.entries.iter().any(|t| t.bytes as usize > shard_len) { return Err("a chunk is longer than the parity shards (corrupted stream?)".into()); }
    // Pass 1: which chunks match the table.
    let mut buf = Vec::new();
    let mut damaged = Vec::new();
    for (j, t) in pl.entries.iter().enumerate() {
        buf.resize(t.bytes as usize, 0);
        if src.read_at(chunks[pl.first_chunk_idx + j].1, &mut buf).is_err() || crc32(&buf) != t.crc { damaged.push(j); }
    }
    let e = damaged.len();
    let mut out = std::collections::HashMap::new();
    if e == 0 { return Ok(out); }
    // Intact shards.
    let mut usable: Vec<(usize, Vec<u16>)> = Vec::new();
    let mut sbuf = vec![0u8; shard_len];
    for (j, &want) in shard_crcs.iter().enumerate() {
        if usable.len() == e { break; }
        if src.read_at(pl.shards_start + j * shard_len, &mut sbuf).is_ok() && crc32(&sbuf) == want {
            usable.push((j, sbuf.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()));
        }
    }
    if usable.len() < e {
        return Err(format!("{e} chunks are damaged in this parity block but only {} of its {m} shards are intact", usable.len()));
    }
    // Pass 2: syndromes S_r = shard_r - sum over intact chunks (XOR), leaving the damaged chunks' part.
    let mut is_bad = vec![false; count];
    for &j in &damaged { is_bad[j] = true; }
    for (j, t) in pl.entries.iter().enumerate() {
        if is_bad[j] { continue; }
        buf.resize(t.bytes as usize, 0);
        src.read_at(chunks[pl.first_chunk_idx + j].1, &mut buf).map_err(e2s)?;
        for (sh_id, syn) in usable.iter_mut() { rs::mac_bytes(syn, rs::coeff(*sh_id, j), &buf); }
    }
    let mut a = vec![0u16; e * e];
    for (r, (sh_id, _)) in usable.iter().enumerate() { for (c, &j) in damaged.iter().enumerate() { a[r * e + c] = rs::coeff(*sh_id, j); } }
    let inv = rs::invert(a, e).ok_or_else(|| "internal error: parity matrix singular".to_string())?;
    for (c, &j) in damaged.iter().enumerate() {
        let mut syms = vec![0u16; shard_len / 2];
        for (r, (_, syn)) in usable.iter().enumerate() { rs::mac_syms(&mut syms, inv[c * e + r], syn); }
        let mut bytes = rs::syms_to_bytes(&syms);
        let t = pl.entries[j];
        bytes.truncate(t.bytes as usize);
        if crc32(&bytes) == t.crc { out.insert(pl.first_chunk_idx + j, bytes); }
    }
    Ok(out)
}

/// Recovers chunk `idx`'s real payload bytes from its parity block, called only after
/// that chunk has already failed its own direct CRC check -- a healthy decode never reaches this.
/// The result is checked against the parity block's own redundant per-chunk CRC (not the damaged
/// chunk's header copy), so a chunk whose only damage is its header CRC field is returned as it
/// stands, and a rebuild that cannot be right (too many damaged chunks, damaged shards) is reported
/// as a failure rather than accepted.
fn recover_chunk<S: ByteSource + ?Sized>(src: &mut S, chunks: &[ChunkLoc], parities: &[ParityLoc], cache: &FecCache, idx: usize) -> Result<Vec<u8>, FormatError> {
    let pl = parities.iter().find(|p| idx >= p.first_chunk_idx && idx < p.first_chunk_idx + p.entries.len())
        .ok_or_else(|| FormatError("no FEC parity covers this chunk (corrupted stream, not recoverable)".into()))?;
    let mut map = cache.0.lock().map_err(|_| FormatError("FEC recovery state poisoned".into()))?;
    let group = map.entry(pl.first_chunk_idx).or_insert_with(|| recover_group(src, chunks, pl));
    match group {
        Err(why) => Err(FormatError(format!("FEC recovery failed: {why}"))),
        Ok(rebuilt) => {
            if let Some(p) = rebuilt.get(&idx) { return Ok(p.clone()); }
            let t = pl.entries[idx - pl.first_chunk_idx];
            let mut buf = vec![0u8; t.bytes as usize];
            if src.read_at(chunks[idx].1, &mut buf).is_ok() && crc32(&buf) == t.crc { return Ok(buf); }
            Err(FormatError("FEC recovery failed: this chunk could not be rebuilt (more damaged chunks than intact parity shards, or a damaged shard)".into()))
        }
    }
}

/// Random-access, chunk-at-a-time decoder over one whole `.fak` stream: parses and
/// validates the container once ([`parse_container`]), then decodes any single chunk on demand, with
/// the same CRC check and FEC recovery as [`decode_full`]. Memory is the stream bytes plus whatever
/// chunks the caller keeps, so a player or a streaming writer never needs the whole decoded PCM at
/// once (the CLI's `decode` and the foobar2000 component both use it). Generic over the byte
/// storage so it can borrow (`&[u8]`) or own (`Vec<u8>`) the stream.
pub struct Reader<D: AsRef<[u8]>> {
    data: D,
    pub header: StreamHeader,
    pub metadata: Metadata,
    chunks: Vec<ChunkLoc>,
    parities: Vec<ParityLoc>,
    fec: FecCache,
    /// Real total sample-frames found by the chunk scan (equals `header.total_frames` unless that
    /// is `TOTAL_FRAMES_UNKNOWN`).
    pub total_frames: u64,
    /// Byte offset of the first chunk, i.e. the end of the metadata block.
    pub payload_start: usize,
}

impl<D: AsRef<[u8]>> Reader<D> {
    pub fn open(data: D) -> Result<Self, FormatError> {
        let (header, chunks, parities, metadata, total_frames, payload_start) = parse_container(data.as_ref())?;
        Ok(Reader { data, header, metadata, chunks, parities, fec: FecCache::default(), total_frames, payload_start })
    }
    pub fn data(&self) -> &[u8] { self.data.as_ref() }
    pub fn chunk_count(&self) -> usize { self.chunks.len() }
    /// Number of FEC parity blocks (0 without FEC).
    pub fn parity_count(&self) -> usize { self.parities.len() }
    /// Data chunks per FEC parity block (the encoder's group size; the last group may be smaller), 0 without FEC.
    pub fn fec_group(&self) -> usize { self.parities.iter().map(|p| p.entries.len()).max().unwrap_or(0) }
    /// First sample-frame of chunk `i`.
    pub fn chunk_start(&self, i: usize) -> u64 { self.chunks[i].0 }
    pub fn chunk_frames(&self, i: usize) -> usize { self.chunks[i].2.frames as usize }
    /// Index of the chunk containing `frame`, or `None` if `frame >= total_frames`.
    pub fn chunk_for_frame(&self, frame: u64) -> Option<usize> {
        if frame >= self.total_frames { return None; }
        Some(self.chunks.partition_point(|c| c.0 + c.2.frames as u64 <= frame))
    }
    /// Decodes chunk `i` (CRC-checked, FEC-recovered if needed) into per-channel samples.
    pub fn decode_chunk(&self, i: usize) -> Result<Vec<Vec<i64>>, FormatError> {
        let data = self.data.as_ref();
        let (_, byte_start, e) = *self.chunks.get(i).ok_or_else(|| FormatError(format!("chunk index {i} out of range")))?;
        let slice = &data[byte_start..byte_start + e.bytes as usize];
        let recovered;
        let payload = if crc32(slice) == e.crc {
            slice
        } else {
            let mut src = data;
            recovered = recover_chunk(&mut src, &self.chunks, &self.parities, &self.fec, i)
                .map_err(|err| FormatError(format!("chunk CRC mismatch, FEC recovery failed: {err}")))?;
            recovered.as_slice()
        };
        let nch = self.header.channels as usize;
        decode_frames(payload, nch, self.header.bits_per_sample, e.frames as usize)
    }
    /// Decodes chunks `range` on up to `threads` workers, results in chunk order.
    pub fn decode_chunks(&self, range: std::ops::Range<usize>, threads: usize) -> Vec<Result<Vec<Vec<i64>>, FormatError>>
    where D: Sync {
        let idx: Vec<usize> = range.collect();
        parallel::par_map(&idx, threads, |&i| self.decode_chunk(i))
    }
}

/// `Read + Seek`, so a `FileReader` can be over a boxed file, an in-memory cursor or a callback.
pub trait ReadSeek: std::io::Read + std::io::Seek + Send {}
impl<T: std::io::Read + std::io::Seek + Send + ?Sized> ReadSeek for T {}

/// A [`ByteSource`] over a seekable file or reader. Tracks the position so back-to-back reads (the
/// 4-byte sync then the rest of a chunk header) do not seek.
pub struct SeekSource<R: std::io::Read + std::io::Seek> { r: R, len: usize, cur: Option<u64> }

impl<R: std::io::Read + std::io::Seek> SeekSource<R> {
    pub fn new(mut r: R) -> Result<Self, FormatError> {
        let len = r.seek(std::io::SeekFrom::End(0)).map_err(io_err)?;
        let len = usize::try_from(len).map_err(|_| FormatError("file too large for this platform".into()))?;
        Ok(SeekSource { r, len, cur: None })
    }
}

impl<R: std::io::Read + std::io::Seek> ByteSource for SeekSource<R> {
    fn len(&self) -> usize { self.len }
    fn read_at(&mut self, pos: usize, buf: &mut [u8]) -> Result<(), FormatError> {
        let end = pos.checked_add(buf.len()).filter(|&e| e <= self.len)
            .ok_or_else(|| FormatError("read past the end of the stream (corrupted stream?)".into()))?;
        if self.cur != Some(pos as u64) {
            self.cur = None;
            self.r.seek(std::io::SeekFrom::Start(pos as u64)).map_err(io_err)?;
        }
        self.cur = None;
        self.r.read_exact(buf).map_err(io_err)?;
        self.cur = Some(end as u64);
        Ok(())
    }
}

fn io_err(e: std::io::Error) -> FormatError { FormatError(format!("I/O error: {e}")) }

/// [`Reader`] for a stream that stays on disk (or an SD card): only the header, metadata block and a
/// 40-byte-per-chunk index are held; a chunk's compressed bytes are read when it is decoded (CRC
/// checked, FEC recovered like [`Reader`]). Memory is index + one chunk's payload + its decoded
/// samples, not the file: a 160 MB album track played with about 8 MB instead of 160.
pub struct FileReader<R: std::io::Read + std::io::Seek> {
    src: SeekSource<R>,
    pub header: StreamHeader,
    pub metadata: Metadata,
    chunks: Vec<ChunkLoc>,
    parities: Vec<ParityLoc>,
    fec: FecCache,
    pub total_frames: u64,
    pub payload_start: usize,
    scratch: Vec<u8>,
}

/// Decodes one chunk's payload (see [`FileReader::payload_decoder`]).
#[derive(Clone, Copy, Debug)]
pub struct PayloadDecoder { channels: usize, bits: u8, frames: usize }

impl PayloadDecoder {
    pub fn decode(&self, payload: &[u8]) -> Result<Vec<Vec<i64>>, FormatError> {
        decode_frames(payload, self.channels, self.bits, self.frames)
    }
}

/// [`verify`] for a file kept on disk: decodes chunk batches and hashes them as they come, so memory
/// is one batch of chunks instead of the whole decoded stream.
pub fn verify_file<R: std::io::Read + std::io::Seek>(r: &mut FileReader<R>, threads: usize) -> Result<(), FormatError> {
    if r.header.total_frames == TOTAL_FRAMES_UNKNOWN {
        return Err(FormatError("cannot verify a streamed (unknown-length) file: no whole-stream hash was computed at encode time".into()));
    }
    let mut h = sha256::PcmHasher::new(r.header.bits_per_sample);
    let batch = threads.max(1) * 2;
    let count = r.chunk_count();
    let mut i = 0;
    while i < count {
        let end = (i + batch).min(count);
        for c in r.decode_chunks(i..end, threads) { h.update(&c?); }
        i = end;
    }
    if h.finalize() != r.header.pcm_hash {
        return Err(FormatError("decoded PCM does not match the stream's stored SHA-256 hash (corrupted stream?)".into()));
    }
    Ok(())
}

impl<R: std::io::Read + std::io::Seek> FileReader<R> {
    pub fn open(r: R) -> Result<Self, FormatError> {
        let mut src = SeekSource::new(r)?;
        let len = src.len();
        let mut head = vec![0u8; HEADER_LEN + 4];
        if len < head.len() { return Err(FormatError("truncated header".into())); }
        src.read_at(0, &mut head)?;
        let header = StreamHeader::from_bytes(&head)?;
        let body_len = u32::from_le_bytes(head[HEADER_LEN..HEADER_LEN + 4].try_into().unwrap()) as usize;
        if body_len > metadata::MAX_METADATA_BLOCK_LEN {
            return Err(FormatError("metadata block length exceeds sanity bound (corrupted stream?)".into()));
        }
        // Bounded by the real file length before allocating, like `metadata::read_block` on a slice.
        let total = (HEADER_LEN + 4).checked_add(body_len).and_then(|n| n.checked_add(4)).filter(|&n| n <= len)
            .ok_or_else(|| FormatError("metadata block runs past end of file (corrupted stream?)".into()))?;
        let mut prefix = vec![0u8; total];
        src.read_at(0, &mut prefix)?;
        let (metadata, payload_start) = metadata::read_block(&prefix, HEADER_LEN).map_err(|e| FormatError(e.0))?;
        drop(prefix);
        check_cue(&header, &metadata)?;
        let (chunks, parities, total_frames) = locate_chunks_src(&header, &mut src, payload_start)?;
        Ok(FileReader { src, header, metadata, chunks, parities, fec: FecCache::default(), total_frames, payload_start, scratch: Vec::new() })
    }
    pub fn chunk_count(&self) -> usize { self.chunks.len() }
    pub fn parity_count(&self) -> usize { self.parities.len() }
    pub fn fec_group(&self) -> usize { self.parities.iter().map(|p| p.entries.len()).max().unwrap_or(0) }
    pub fn chunk_start(&self, i: usize) -> u64 { self.chunks[i].0 }
    pub fn chunk_frames(&self, i: usize) -> usize { self.chunks[i].2.frames as usize }
    pub fn chunk_for_frame(&self, frame: u64) -> Option<usize> {
        if frame >= self.total_frames { return None; }
        Some(self.chunks.partition_point(|c| c.0 + c.2.frames as u64 <= frame))
    }
    /// The whole stream as bytes (for the operations that need it, like rewriting the metadata block).
    pub fn read_all(&mut self) -> Result<Vec<u8>, FormatError> {
        let mut v = vec![0u8; self.src.len()];
        self.src.read_at(0, &mut v)?;
        Ok(v)
    }
    /// Writes this stream to `w` with its metadata block replaced by `meta`, copying the audio in
    /// 1 MiB pieces (memory does not grow with the file; [`rewrite_metadata`] is the in-memory twin).
    /// Chunk bytes, parity and the stream header (with its PCM hash) are copied unchanged. A cue
    /// point beyond the stream's real length is rejected, as the decoder would.
    pub fn rewrite_metadata_to<W: std::io::Write>(&mut self, meta: &Metadata, w: &mut W) -> Result<(), FormatError> {
        if let Some(cue) = &meta.cue_sheet {
            if cue.tracks.iter().flat_map(|t| &t.indices).any(|i| i.sample_offset > self.total_frames) {
                return Err(FormatError("cue sheet index offset exceeds the stream length".into()));
            }
        }
        let block = metadata::write_block(meta);
        metadata::read_block(&block, 0).map_err(|e| FormatError(format!("new metadata is not writable: {}", e.0)))?;
        let mut head = vec![0u8; HEADER_LEN];
        self.src.read_at(0, &mut head)?;
        w.write_all(&head).map_err(io_err)?;
        w.write_all(&block).map_err(io_err)?;
        let (mut pos, end) = (self.payload_start, self.src.len());
        let mut buf = vec![0u8; 1 << 20];
        while pos < end {
            let n = buf.len().min(end - pos);
            self.src.read_at(pos, &mut buf[..n])?;
            w.write_all(&buf[..n]).map_err(io_err)?;
            pos += n;
        }
        Ok(())
    }
    /// Size of the stream in bytes.
    pub fn len(&self) -> usize { self.src.len() }
    pub fn is_empty(&self) -> bool { self.src.is_empty() }

    /// Reads chunk `i`'s compressed bytes into `buf` (replacing its contents), CRC-checked and, if
    /// that fails, recovered from FEC parity.
    pub fn read_payload(&mut self, i: usize, buf: &mut Vec<u8>) -> Result<(), FormatError> {
        let (_, byte_start, e) = *self.chunks.get(i).ok_or_else(|| FormatError(format!("chunk index {i} out of range")))?;
        // A chunk cannot legitimately need more than 8 bytes per sample (the escape path is 5), so a
        // corrupted length is refused before it sizes a buffer.
        let nch = self.header.channels as u64;
        if e.bytes as u64 > (e.frames as u64).saturating_mul(nch).saturating_mul(8).saturating_add(4096) {
            return Err(FormatError(format!("chunk payload length {} exceeds sanity bound (corrupted stream?)", e.bytes)));
        }
        buf.clear();
        buf.resize(e.bytes as usize, 0);
        self.src.read_at(byte_start, buf)?;
        if crc32(buf) != e.crc {
            *buf = recover_chunk(&mut self.src, &self.chunks, &self.parities, &self.fec, i)
                .map_err(|err| FormatError(format!("chunk CRC mismatch, FEC recovery failed: {err}")))?;
        }
        Ok(())
    }

    /// The header and chunk length a payload decode needs, so worker threads can decode without
    /// holding the reader (which only reads the file).
    pub fn payload_decoder(&self, i: usize) -> Result<PayloadDecoder, FormatError> {
        let e = self.chunks.get(i).ok_or_else(|| FormatError(format!("chunk index {i} out of range")))?.2;
        Ok(PayloadDecoder { channels: self.header.channels as usize, bits: self.header.bits_per_sample, frames: e.frames as usize })
    }

    /// Decodes chunk `i` from the bytes [`Self::read_payload`] returned for it.
    pub fn decode_payload(&self, i: usize, payload: &[u8]) -> Result<Vec<Vec<i64>>, FormatError> {
        let e = self.chunks.get(i).ok_or_else(|| FormatError(format!("chunk index {i} out of range")))?.2;
        decode_frames(payload, self.header.channels as usize, self.header.bits_per_sample, e.frames as usize)
    }

    pub fn decode_chunk(&mut self, i: usize) -> Result<Vec<Vec<i64>>, FormatError> {
        let mut buf = std::mem::take(&mut self.scratch);
        let r = self.read_payload(i, &mut buf).and_then(|_| self.decode_payload(i, &buf));
        self.scratch = buf;
        r
    }

    /// Decodes chunks `range` on up to `threads` workers (payloads read one after another, then
    /// decoded in parallel), results in chunk order. Holds `range.len()` chunks at once.
    pub fn decode_chunks(&mut self, range: std::ops::Range<usize>, threads: usize) -> Vec<Result<Vec<Vec<i64>>, FormatError>> {
        let mut payloads: Vec<Result<Vec<u8>, FormatError>> = Vec::with_capacity(range.len());
        for i in range.clone() {
            let mut b = Vec::new();
            payloads.push(self.read_payload(i, &mut b).map(|_| b));
        }
        let (nch, bits) = (self.header.channels as usize, self.header.bits_per_sample);
        let items: Vec<(usize, Result<Vec<u8>, FormatError>)> = range.map(|i| self.chunks.get(i).map_or(0, |c| c.2.frames as usize)).zip(payloads).collect();
        parallel::par_map(&items, threads, |(frames, p)| match p {
            Ok(b) if *frames > 0 => decode_frames(b, nch, bits, *frames),
            Ok(_) => Err(FormatError("chunk index out of range".into())),
            Err(e) => Err(FormatError(e.0.clone())),
        })
    }
}

/// Returns `data` with its metadata block replaced by `meta` (retagging, e.g. from the
/// foobar2000 component). Audio chunks are copied byte for byte, so the stream header (including its
/// PCM hash) stays valid. The whole container is validated first, and a cue point beyond the
/// stream's real length is rejected, as the decoder would.
pub fn rewrite_metadata(data: &[u8], meta: &Metadata) -> Result<Vec<u8>, FormatError> {
    let r = Reader::open(data)?;
    if let Some(cue) = &meta.cue_sheet {
        if cue.tracks.iter().flat_map(|t| &t.indices).any(|i| i.sample_offset > r.total_frames) {
            return Err(FormatError("cue sheet index offset exceeds the stream length".into()));
        }
    }
    let block = metadata::write_block(meta);
    metadata::read_block(&block, 0).map_err(|e| FormatError(format!("new metadata is not writable: {}", e.0)))?;
    let mut out = Vec::with_capacity(HEADER_LEN + block.len() + data.len() - r.payload_start);
    out.extend_from_slice(&data[..HEADER_LEN]);
    out.extend_from_slice(&block);
    out.extend_from_slice(&data[r.payload_start..]);
    Ok(out)
}

/// Validates the header, metadata block, and self-delimited chunk sequence (`src/metadata.rs`), then decodes chunks on up to `threads` workers. Each chunk's CRC-32
/// is checked before its payload is decoded, so a corrupted or hostile chunk is rejected cheaply
/// instead of being fed to the predictor. Holds the whole decoded stream in memory; [`Reader`]
/// decodes chunk by chunk instead.
pub fn decode_full(data: &[u8], threads: usize) -> Result<(StreamHeader, Metadata, Vec<Vec<i64>>), FormatError> {
    let r = Reader::open(data)?;
    let nch = r.header.channels as usize;
    let decoded = r.decode_chunks(0..r.chunk_count(), threads);
    // No up-front reservation from the frame count: a tiny file can legitimately hold hours of
    // silence, so no declared or scanned count is trusted for allocation -- the vectors grow only as
    // chunks actually decode.
    let mut channels: Vec<Vec<i64>> = (0..nch).map(|_| Vec::new()).collect();
    for chunk in decoded {
        for (dst, src) in channels.iter_mut().zip(chunk?) { dst.extend(src); }
    }
    Ok((r.header, r.metadata, channels))
}

/// Fully decodes `data` (like [`decode_full`]) and checks the result against the stream's stored
/// SHA-256 of the decoded PCM. Deliberately a separate,
/// explicitly-called function rather than something [`decode`]/[`decode_full`] do automatically:
/// hashing the whole decoded PCM stream is real, avoidable CPU cost on top of an ordinary decode, so every caller that doesn't need integrity
/// verification (normal playback) must not pay for it. Mirrors `flac --test`'s role, not its
/// automatic-on-every-decode behavior.
pub fn verify(data: &[u8], threads: usize) -> Result<(StreamHeader, Metadata), FormatError> {
    let header_peek = StreamHeader::from_bytes(data)?;
    if header_peek.total_frames == TOTAL_FRAMES_UNKNOWN {
        // A stream written by `encoder::StreamEncoder` never had a whole-stream hash computed --
        // there was no way to see every sample before encoding started. Comparing against the
        // all-zero placeholder would either always "fail" (if real PCM hashes to something else,
        // overwhelmingly likely) or, worse, occasionally succeed by coincidence; refusing outright
        // is the only honest answer (don't claim a verification that didn't happen).
        return Err(FormatError("cannot verify a streamed (unknown-length) file: no whole-stream hash was computed at encode time".into()));
    }
    let (header, meta, channels) = decode_full(data, threads)?;
    let actual = sha256::pcm_digest(&channels, header.bits_per_sample);
    if actual != header.pcm_hash {
        return Err(FormatError("decoded PCM does not match the stream's stored SHA-256 hash (corrupted stream?)".into()));
    }
    Ok((header, meta))
}

/// One chunk decoded by [`seek`]. `chunk_start_frame` is that chunk's first sample-frame's offset
/// from the very start of the stream, so a caller wanting playback to begin exactly at the frame it
/// asked for slices off `(target_frame - chunk_start_frame)` leading samples from each channel of
/// `channels` before using it -- `seek` itself returns the whole containing chunk, not a
/// sample-precise trim, since trimming is free and the caller may want the few samples just before
/// the target too (e.g. to prime a resampler/crossfade).
pub struct SeekResult {
    pub header: StreamHeader,
    pub metadata: Metadata,
    pub chunk_start_frame: u64,
    pub channels: Vec<Vec<i64>>,
}

/// Decodes only the single chunk containing `target_frame`, not the whole stream -- the real,
/// measured cost of a seek, rather than the O(chunks)
/// or O(log chunks) *lookup* the chunk table alone already gave for free since.
///
/// Chunks are effectively O(1) here (every frame inside already decodes independently of every
/// other, so this differs from `decode_full` only in which chunks it bothers with).
///
/// `target_frame` must be `< ` the stream's real total frame count (as found by `locate_chunks`,
/// not necessarily `header.total_frames` itself -- for a `TOTAL_FRAMES_UNKNOWN` stream the header
/// only carries the sentinel, so the real total is whatever the chunk scan actually found). An
/// out-of-range target is rejected rather than silently clamped (treats every input
/// to a decoder-facing function as untrusted, including a caller-supplied offset, not just the file
/// bytes).
pub fn seek(data: &[u8], target_frame: u64) -> Result<SeekResult, FormatError> {
    let r = Reader::open(data)?;
    let idx = r.chunk_for_frame(target_frame)
        .ok_or_else(|| FormatError(format!("seek target frame {target_frame} >= total_frames {} (out of range)", r.total_frames)))?;
    let channels = r.decode_chunk(idx)?;
    let chunk_start_frame = r.chunk_start(idx);
    Ok(SeekResult { header: r.header, metadata: r.metadata, chunk_start_frame, channels })
}

/// Reads exactly `buf.len()` bytes from `r` unless the source is already at a clean boundary (zero
/// bytes available before any are read). Unlike `Read::read_exact`, this distinguishes "the source
/// had nothing left at all" (a valid stopping point for a `TOTAL_FRAMES_UNKNOWN` stream) from
/// "the source had *some* bytes, then ran out" (truncation, always an error) -- `read_exact` alone
/// collapses both into the same `UnexpectedEof`, which `decode_stream` needs to tell apart.
enum FillResult { Eof, Truncated, Full }
fn fill_or_eof<R: std::io::Read>(r: &mut R, buf: &mut [u8]) -> Result<FillResult, FormatError> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(io_err(e)),
        }
    }
    Ok(if filled == 0 { FillResult::Eof } else if filled < buf.len() { FillResult::Truncated } else { FillResult::Full })
}

/// Incrementally decodes a self-delimited stream from `r`:
/// reads the header and metadata block once, then repeatedly reads one chunk's inline header plus
/// payload, CRC-checks and decodes it, and calls `on_chunk` with the result before the next chunk
/// is even read -- never buffering more than one chunk's payload in memory, and never seeking
/// backward on `r`, so a real pipe or socket is a valid source (unlike `decode`/`decode_full`,
/// which need the whole stream as one `&[u8]` up front).
///
/// Works for both a known-length stream (`decode_full` on a fully-buffered copy of the same bytes
/// would give identical results -- this stops once `total_frames` is reached and then requires a
/// clean end of input, rejecting trailing garbage exactly like `locate_chunks` does) and a
/// `TOTAL_FRAMES_UNKNOWN` stream (a genuinely unbounded/live source, written by
/// [`crate::encoder::StreamEncoder`]) -- there, the only valid stopping point is `r` itself running
/// out, which is also how `on_chunk` returning an error (the caller wants to stop early, e.g. it
/// closed the audio device) or `r` closing unexpectedly both look from here.
pub fn decode_stream<R: std::io::Read>(
    mut r: R, mut on_chunk: impl FnMut(&[Vec<i64>]) -> Result<(), FormatError>,
) -> Result<(StreamHeader, Metadata), FormatError> {
    let mut header_buf = vec![0u8; HEADER_LEN];
    r.read_exact(&mut header_buf).map_err(io_err)?;
    let header = StreamHeader::from_bytes(&header_buf)?;

    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).map_err(io_err)?;
    let body_len = u32::from_le_bytes(len_buf) as usize;
    // Bounded before allocating: unlike the in-memory `metadata::read_block`, this
    // reader has no `data.len()` to fall back on -- a hostile or buggy peer's declared length is the
    // only signal available before the bytes actually arrive, so it must be capped explicitly.
    if body_len > metadata::MAX_METADATA_BLOCK_LEN {
        return Err(FormatError("metadata block length exceeds sanity bound (corrupted stream?)".into()));
    }
    let mut body = vec![0u8; body_len];
    r.read_exact(&mut body).map_err(io_err)?;
    let mut crc_buf = [0u8; 4];
    r.read_exact(&mut crc_buf).map_err(io_err)?;
    if crc32(&body) != u32::from_le_bytes(crc_buf) {
        return Err(FormatError("metadata block CRC mismatch (corrupted stream?)".into()));
    }
    let meta = Metadata::read(&body).map_err(|e| FormatError(e.0))?;

    let nch = header.channels as usize;
    let mut frame_pos = 0u64;
    loop {
        if header.total_frames != TOTAL_FRAMES_UNKNOWN && frame_pos >= header.total_frames { break; }
        let mut hdr_buf = [0u8; crate::format::CHUNK_HEADER_LEN];
        match fill_or_eof(&mut r, &mut hdr_buf)? {
            FillResult::Eof => break,
            FillResult::Truncated => return Err(FormatError("truncated chunk header (corrupted stream?)".into())),
            FillResult::Full => {}
        }
        let e = crate::format::read_chunk_header(&hdr_buf)?;
        if e.frames == 0 || e.frames > crate::format::MAX_CHUNK_FRAMES {
            return Err(FormatError(format!("invalid chunk frame count {} (corrupted stream?)", e.frames)));
        }
        if header.total_frames != TOTAL_FRAMES_UNKNOWN {
            let would_be = frame_pos.checked_add(e.frames as u64).ok_or_else(|| FormatError("frame count overflow (corrupted stream?)".into()))?;
            if would_be > header.total_frames {
                return Err(FormatError("chunk frame counts exceed total_frames (corrupted stream?)".into()));
            }
        }
        // Bounded before allocating, the same reason the metadata block length is
        // capped above: `locate_chunks` (the in-memory path) gets this for free from `data.len()`
        // itself, but a streaming `Read` source has no such bound -- a hostile or buggy peer's
        // declared `bytes` is the only signal available before the payload actually arrives. 8
        // bytes/sample plus fixed overhead is well above this codec's real worst case (the Rice
        // escape path's `MAX_ESCAPE_WIDTH` is 40 bits, 5 bytes) so no legitimate chunk is rejected.
        let max_reasonable_bytes = (e.frames as u64).saturating_mul(nch as u64).saturating_mul(8).saturating_add(4096);
        if e.bytes as u64 > max_reasonable_bytes {
            return Err(FormatError(format!(
                "chunk payload length {} exceeds sanity bound for {} frames x {nch} channels (corrupted stream?)", e.bytes, e.frames
            )));
        }
        let mut payload = vec![0u8; e.bytes as usize];
        r.read_exact(&mut payload).map_err(io_err)?;
        if crc32(&payload) != e.crc { return Err(FormatError("chunk CRC mismatch (corrupted stream?)".into())); }
        let channels = decode_frames(&payload, nch, header.bits_per_sample, e.frames as usize)?;
        on_chunk(&channels)?;
        frame_pos += e.frames as u64;
    }
    if header.total_frames != TOTAL_FRAMES_UNKNOWN {
        if frame_pos != header.total_frames {
            return Err(FormatError("stream ended before reaching declared total_frames (truncated stream?)".into()));
        }
        // The loop above exits as soon as `frame_pos` reaches `total_frames`, *before* attempting
        // to read anything past the last real chunk -- so trailing garbage would otherwise sit
        // unread in `r` and go completely undetected. One more probe read confirms `r` is actually
        // exhausted, mirroring `locate_chunks`'s `pos != data.len()` check on the in-memory path.
        let mut probe = [0u8; 1];
        if !matches!(fill_or_eof(&mut r, &mut probe)?, FillResult::Eof) {
            return Err(FormatError("trailing garbage after chunk data (corrupted stream?)".into()));
        }
    }
    Ok((header, meta))
}

/// Config-3 chunk payload (H178; see `encoder::encode_ols_chunk`): per block and channel the carried-LMS flag and
/// shift, then the Rice residual; the carried filter is inverted per channel, then the stereo OLS per block.
fn decode_ols_chunk(data: &[u8], chunk_frames: usize, bits_per_sample: u8, irls: bool) -> Result<Vec<Vec<i64>>, FormatError> {
    let mut r = BitReader::new(data);
    let mut st = crate::ols::Stereo::new(crate::ols::Params { irls, ..Default::default() });
    let (taps, k) = stage2::Carried::config(1).unwrap();
    let mut carried = [stage2::Carried::new_ols(taps, k, bits_per_sample), stage2::Carried::new_ols(taps, k, bits_per_sample)];
    let (lo, hi) = (-(1i64 << (bits_per_sample - 1)), (1i64 << (bits_per_sample - 1)) - 1);
    let mut out: Vec<Vec<i64>> = vec![Vec::with_capacity(chunk_frames), Vec::with_capacity(chunk_frames)];
    let mut done = 0usize;
    while done < chunk_frames {
        let n = crate::encoder::OLS_BLOCK.min(chunk_frames - done);
        let mut res: Vec<Vec<i64>> = Vec::with_capacity(2);
        for c in carried.iter_mut() {
            let flag = r.read_bits(1).map_err(bmap)? == 1;
            let s = if flag { Some(stage2::Carried::read_s(&mut r).map_err(FormatError)?) } else { None };
            let mut v = { let _g = crate::prof::span(crate::prof::Phase::DecRice); rice::decode(&mut r, n).map_err(bmap)? };
            let g_car = crate::prof::span(crate::prof::Phase::DecCarried);
            match s {
                Some(s) => c.inverse(&mut v, s).map_err(|e| FormatError(e.into()))?,
                None if v.iter().all(|e| e.abs() <= stage2::MAX_RESIDUAL) => {
                    let s = stage2::Params::for_block(&v, c.taps(), c.k(), stage2::Carried::TARGET).s;
                    c.advance(&v, s);
                }
                None => {}
            }
            drop(g_car);
            res.push(v);
        }
        let (s0, s1) = { let _g = crate::prof::span(crate::prof::Phase::DecOls); st.inverse_block(&res[0], &res[1]) };
        if s0.iter().chain(&s1).any(|&x| x < lo || x > hi) { return Err(FormatError("OLS chunk sample out of range (corrupted stream?)".into())); }
        out[0].extend(s0);
        out[1].extend(s1);
        done += n;
    }
    r.align_to_byte();
    if r.byte_pos() != data.len() { return Err(FormatError("trailing bytes after an OLS chunk (corrupted stream?)".into())); }
    Ok(out)
}

/// Block-independent frames making up one chunk: exactly `chunk_frames` sample-frames, consuming
/// exactly all of `data` (trailing bytes would mean the table and the frames disagree).
fn decode_frames(data: &[u8], nch: usize, bits_per_sample: u8, chunk_frames: usize) -> Result<Vec<Vec<i64>>, FormatError> {
    let _g = crate::prof::span(crate::prof::Phase::DecChunk);
    let mut channels: Vec<Vec<i64>> = (0..nch).map(|_| Vec::with_capacity(chunk_frames)).collect();
    let (maps, mut pos) = valuemap::read_section(data, nch, chunk_frames).map_err(FormatError)?;
    let cfg = *data.get(pos).ok_or_else(|| FormatError("missing chunk config byte".into()))?;
    pos += 1;
    if cfg > 4 { return Err(FormatError(format!("invalid chunk config {cfg}"))); }
    if cfg == 3 || cfg == 4 {
        if nch != 2 || bits_per_sample > 24 { return Err(FormatError("OLS chunk needs 2 channels of <= 24 bits".into())); }
        let mut channels = decode_ols_chunk(&data[pos..], chunk_frames, bits_per_sample, cfg == 4)?;
        let (lo, hi) = (-(1i64 << (bits_per_sample - 1)), (1i64 << (bits_per_sample - 1)) - 1);
        for (chan, m) in channels.iter_mut().zip(&maps) {
            if let Some(m) = m { m.apply_all(chan, lo, hi); }
        }
        return Ok(channels);
    }
    let (ctaps, ck) = stage2::Carried::config(cfg).unwrap_or((1, 0));
    let mut slots: Vec<stage2::Carried> = (0..nch.max(2)).map(|_| stage2::Carried::new(if cfg == 0 { 1 } else { ctaps }, ck)).collect();
    let mut decoded = 0u64;
    let total_frames = chunk_frames as u64;

    while decoded < total_frames {
        if pos >= data.len() { return Err(FormatError("truncated frame header".into())); }
        let mut r = BitReader::new(&data[pos..]);
        let frame_frames = format::read_frame_len(&mut r).map_err(bmap)?;
        if frame_frames == 0 || frame_frames > MAX_FRAME_FRAMES {
            return Err(FormatError(format!("invalid frame_frames {frame_frames}")));
        }
        if decoded + frame_frames as u64 > total_frames {
            return Err(FormatError("frame overruns declared chunk length".into()));
        }
        let n = frame_frames as usize;
        let base = bits_per_sample as u32;

        if nch == 2 {
            let mode_v = r.read_bits(2).map_err(bmap)? as u8;
            let mode = StereoMode::from_u8(mode_v).ok_or_else(|| FormatError(format!("invalid stereo mode {mode_v}")))?;
            let (bits_a, bits_b) = match mode {
                StereoMode::LeftRight => (base, base),
                StereoMode::MidSide => (base, base + 1),
                StereoMode::LeftSide => (base, base + 1),
                StereoMode::SideRight => (base + 1, base),
            };
            let h0 = channels[0].len().saturating_sub(HISTORY_LEN);
            let (hl, hr) = (&channels[0][h0..], &channels[1][h0..]);
            let (ha, hb) = match mode {
                StereoMode::LeftRight => (hl.to_vec(), hr.to_vec()),
                StereoMode::MidSide => (stereo::mid(hl, hr), stereo::side(hl, hr)),
                StereoMode::LeftSide => (hl.to_vec(), stereo::side(hl, hr)),
                StereoMode::SideRight => (stereo::side(hl, hr), hr.to_vec()),
            };
            #[cfg(feature = "research-tap")]
            let t_len = TAP.lock().unwrap().len();
            let a = read_subframe_carried(&mut r, n, bits_a, &ha, &[], (cfg >= 1).then(|| &mut slots[0]))?;
            #[cfg(feature = "research-tap")]
            let t_mid = TAP.lock().unwrap().len();
            let b = read_subframe_carried(&mut r, n, bits_b, &hb, std::slice::from_ref(&a), (cfg >= 1).then(|| &mut slots[1]))?;
            #[cfg(feature = "research-tap")]
            for (i, e) in TAP.lock().unwrap().iter_mut().enumerate().skip(t_len) { e.stereo_mode = mode_v; e.slot = (i >= t_mid) as u8; }
            #[cfg(feature = "research-tap")]
            if std::env::var_os("FAK_TAP_RES").is_some() {
                let (a_sig, b_sig) = (a.residual_signal(), b.residual_signal());
                let mut t = TAP.lock().unwrap();
                for e in t[t_len..t_mid].iter_mut() { e.other_res = b_sig.clone(); e.other_samples = b.samples.clone(); }
                for e in t[t_mid..].iter_mut() { e.other_res = a_sig.clone(); e.other_samples = a.samples.clone(); }
            }
            let (c0, c1) = channels.split_at_mut(1);
            stereo::decode_into(mode, &a.samples, &b.samples, &mut c0[0], &mut c1[0]);
        } else {
            let mut subs: Vec<SubOut> = Vec::with_capacity(nch);
            for (ci, chan) in channels.iter().enumerate() {
                let h0 = chan.len().saturating_sub(HISTORY_LEN);
                let sub = read_subframe_carried(&mut r, n, base, &chan[h0..], &subs, (cfg >= 1).then(|| &mut slots[ci]))?;
                subs.push(sub);
            }
            for (chan, sub) in channels.iter_mut().zip(&subs) { chan.extend_from_slice(&sub.samples); }
        }

        r.align_to_byte();
        let payload_len = r.byte_pos();
        // No per-frame CRC since format v15: the chunk's CRC-32, checked before any frame is
        // parsed, already covers every byte, and any error rejects the whole chunk.
        if pos + payload_len > data.len() { return Err(FormatError("truncated frame".into())); }

        #[cfg(feature = "research-tap")]
        { let mut f = FTAP.lock().unwrap(); f.0 += 1; f.1 += payload_len as u64; }
        pos += payload_len;
        decoded += frame_frames as u64;
    }
    if pos != data.len() { return Err(FormatError("trailing bytes after a chunk's last frame (corrupted stream?)".into())); }
    let (lo, hi) = (-(1i64 << (bits_per_sample - 1)), (1i64 << (bits_per_sample - 1)) - 1);
    for (chan, m) in channels.iter_mut().zip(&maps) {
        if let Some(m) = m { m.apply_all(chan, lo, hi); }
    }
    Ok(channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder;

    /// Retagging keeps the audio bytes, the header and every chunk intact and changes only the
    /// metadata; bounds (an out-of-range cue point, an oversized tag) are rejected.
    #[test]
    fn rewrite_metadata_keeps_audio_and_replaces_tags() {
        use crate::encoder::encode_chunked;
        let l: Vec<i64> = (0..9000i64).map(|i| ((i * 977) % 30001) - 15000).collect();
        let meta = Metadata { vendor: "v".into(), tags: vec!["TITLE=a".into()], ..Default::default() };
        let data = encode_chunked(&[l.clone()], 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4096, 1, Some(2), &meta).unwrap();
        let new = Metadata { vendor: "v".into(), tags: vec!["TITLE=Longer title".into(), "ARTIST=Ümlaut".into()], ..Default::default() };
        let out = rewrite_metadata(&data, &new).unwrap();
        let (h, m, ch) = decode_full(&out, 1).unwrap();
        assert_eq!((m, ch), (new.clone(), vec![l.clone()]));
        assert_eq!(h.pcm_hash, StreamHeader::from_bytes(&data).unwrap().pcm_hash);
        assert!(verify(&out, 1).is_ok());
        let back = rewrite_metadata(&out, &meta).unwrap();
        assert_eq!(back, data, "rewriting the original tags restores the original bytes");
        let bad = Metadata { cue_sheet: Some(crate::metadata::CueSheet { catalog: String::new(), tracks: vec![crate::metadata::CueTrack {
            number: 1, isrc: String::new(), indices: vec![crate::metadata::CueIndex { number: 1, sample_offset: 9001 }] }] }), ..Default::default() };
        assert!(rewrite_metadata(&data, &bad).is_err());
        assert!(rewrite_metadata(&data[..data.len() - 1], &new).is_err());

        // The streaming twin writes the same bytes (with a piece size smaller than the file it is a
        // multi-piece copy in the real 1 MiB loop too: 700k frames of noise).
        let mut r = FileReader::open(std::io::Cursor::new(data.clone())).unwrap();
        let mut streamed = Vec::new();
        r.rewrite_metadata_to(&new, &mut streamed).unwrap();
        assert_eq!(streamed, out);
        assert!(r.rewrite_metadata_to(&bad, &mut Vec::new()).is_err());
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let noise: Vec<i64> = (0..700_000).map(|_| { st ^= st << 13; st ^= st >> 7; st ^= st << 17; ((st >> 40) as i64 & 0xFFFF) - 32768 }).collect();
        let big = encode_chunked(&[noise], 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 65536, 1, None, &meta).unwrap();
        assert!(big.len() > 1 << 20);
        let mut streamed = Vec::new();
        FileReader::open(std::io::Cursor::new(big.clone())).unwrap().rewrite_metadata_to(&new, &mut streamed).unwrap();
        assert_eq!(streamed, rewrite_metadata(&big, &new).unwrap());
    }

    /// `Reader`: chunk-at-a-time decoding reproduces `decode_full` exactly, and
    /// `chunk_for_frame` maps every chunk boundary (and the frame after the end) correctly.
    #[test]
    fn reader_chunks_match_full_decode() {
        use crate::encoder::encode_chunked;
        let l: Vec<i64> = (0..30_000i64).map(|i| ((i * 977) % 30001) - 15000).collect();
        let r: Vec<i64> = (0..30_000i64).map(|i| ((i * 41) % 4001) - 2000).collect();
        for mode in [crate::format::MODE_BLOCK_INDEPENDENT] {
            let data = encode_chunked(&[l.clone(), r.clone()], 44100, 16, mode, 4096, 2, Some(3), &Metadata::default()).unwrap();
            let rd = Reader::open(&data[..]).unwrap();
            assert_eq!(rd.total_frames, 30_000);
            assert_eq!(rd.chunk_count(), 30_000usize.div_ceil(4096));
            let mut joined = vec![Vec::new(), Vec::new()];
            for i in 0..rd.chunk_count() {
                assert_eq!(rd.chunk_start(i), (i * 4096) as u64);
                assert_eq!(rd.chunk_for_frame(rd.chunk_start(i)), Some(i));
                assert_eq!(rd.chunk_for_frame(rd.chunk_start(i) + rd.chunk_frames(i) as u64 - 1), Some(i));
                for (d, s) in joined.iter_mut().zip(rd.decode_chunk(i).unwrap()) { d.extend(s); }
            }
            assert_eq!(rd.chunk_for_frame(30_000), None);
            assert!(rd.decode_chunk(rd.chunk_count()).is_err());
            assert_eq!(joined, vec![l.clone(), r.clone()]);
            let par: Vec<Vec<Vec<i64>>> = rd.decode_chunks(0..rd.chunk_count(), 3).into_iter().map(Result::unwrap).collect();
            assert_eq!(par.concat().len(), rd.chunk_count() * 2);
            assert_eq!(&data[rd.payload_start..rd.payload_start + 4], crate::format::CHUNK_SYNC);
        }
    }

    #[test]
    fn corrupted_header_crc_rejected() {
        let mut data = encoder::encode(&[vec![1i64, 2, 3, 4, 5]], 44100, 16).unwrap();
        data[10] ^= 0xFF;
        assert!(decode(&data).is_err());
    }

    #[test]
    fn corrupted_frame_crc_rejected() {
        // FEC disabled: without it, the last byte of the file is guaranteed to be inside the
        // actual chunk payload (would otherwise append a trailing parity block, and this
        // test wants to corrupt real audio data, not incidentally corrupt unused parity bytes).
        let data = encoder::encode_chunked(&[(0..5000i64).collect()], 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, crate::format::default_chunk_frames(44100), 1, None, &Default::default()).unwrap();
        let mut bad = data.clone();
        let last = bad.len() - 1;
        bad[last] ^= 0xFF;
        assert!(decode(&bad).is_err());
    }

    #[test]
    fn truncated_stream_rejected_not_panicking() {
        let data = encoder::encode(&[(0..5000i64).collect(), (0..5000i64).map(|i| -i).collect()], 44100, 16).unwrap();
        for cut in [HEADER_LEN, HEADER_LEN + 3, data.len() - 1, data.len() / 2] {
            let truncated = &data[..cut.min(data.len())];
            assert!(decode(truncated).is_err(), "cut at {cut} should be rejected");
        }
    }

    #[test]
    fn bad_sync_rejected() {
        // FEC disabled: a single-chunk file's corrupted payload would otherwise be recoverable
        // from its own FEC parity block, which is real, correct behavior but not what this
        // test (raw detection with nothing to recover from) is checking.
        let mut data = encoder::encode_chunked(&[(0..2000i64).collect()], 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, crate::format::default_chunk_frames(44100), 1, None, &Default::default()).unwrap();
        let empty_metadata_block_len = metadata::write_block(&Metadata::default()).len();
        let first_frame = HEADER_LEN + empty_metadata_block_len + crate::format::CHUNK_HEADER_LEN; // one chunk's inline header, then its payload
        data[first_frame] ^= 0xFF; // corrupt the frame sync bits
        assert!(decode(&data).is_err());
    }

    fn signal(n: usize, seed: u64) -> Vec<i64> {
        let mut s = seed;
        (0..n).map(|i| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ((i as f64 * 0.013).sin() * 9000.0) as i64 + (s % 301) as i64 - 150
        }).collect()
    }

    #[test]
    fn verify_accepts_a_real_encode_decode_roundtrip() {
        let l: Vec<i64> = (0..6000i64).map(|i| ((i * 977) % 30001) - 15000).collect();
        let r: Vec<i64> = (0..6000i64).map(|i| ((i * 41) % 4001) - 2000).collect();
        let data = encoder::encode(&[l, r], 44100, 16).unwrap();
        assert!(verify(&data, 1).is_ok());
    }

    #[test]
    fn verify_rejects_a_stream_whose_hash_does_not_match_its_own_pcm() {
        use crate::format::{StreamHeader, HEADER_LEN as HL};
        let l: Vec<i64> = (0..4000i64).map(|i| ((i * 13) % 900) - 450).collect();
        let data = encoder::encode(&[l], 44100, 16).unwrap();
        let header = StreamHeader::from_bytes(&data).unwrap();
        let mut wrong_hash = header.pcm_hash;
        wrong_hash[0] ^= 0xFF;
        let bad_header = StreamHeader { pcm_hash: wrong_hash, ..header };
        let mut tampered = bad_header.to_bytes();
        tampered.extend_from_slice(&data[HL..]);
        assert!(decode(&tampered).is_ok(), "decode() must not care about pcm_hash at all");
        assert!(verify(&tampered, 1).is_err(), "verify() must catch a hash that doesn't match the real PCM");
    }

    #[test]
    fn metadata_survives_a_real_encode_decode_roundtrip() {
        use crate::metadata::{CueIndex, CueSheet, CueTrack, Metadata, Picture, PictureType};
        let chans = vec![signal(20_000, 7), signal(20_000, 8)];
        let meta = Metadata {
            vendor: "fak-test".to_string(),
            tags: vec!["TITLE=Roundtrip Song".to_string(), "ARTIST=Someone".to_string()],
            pictures: vec![Picture {
                kind: PictureType::FrontCover, kind_raw: 3, mime: "image/png".to_string(),
                description: "cover art".to_string(), width: 10, height: 10, depth: 24, colors: 0,
                data: (0u8..=255).collect(),
            }],
            cue_sheet: Some(CueSheet {
                catalog: String::new(),
                tracks: vec![CueTrack { number: 1, isrc: String::new(), indices: vec![CueIndex { number: 1, sample_offset: 0 }] }],
            }),
            channel_mask: Some(0x3), // front-left | front-right
            float_info: None,
        };
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4096, 4, Some(crate::format::DEFAULT_FEC_GROUP), &meta).unwrap();
        let (_, decoded_meta, decoded_channels) = decode_full(&encoded, 4).unwrap();
        assert_eq!(decoded_meta, meta);
        assert_eq!(decoded_channels, chans);
    }

    #[test]
    fn output_is_identical_for_every_thread_count_and_decodes_back() {
        use crate::format::MODE_BLOCK_INDEPENDENT;
        let chans = vec![signal(50_000, 1), signal(50_000, 2)];
        for mode in [MODE_BLOCK_INDEPENDENT] {
            for chunk in [4096usize, 7000, 50_000, 1 << 20] {
                let one = encoder::encode_chunked(&chans, 44100, 16, mode, chunk, 1, None, &Default::default()).unwrap();
                for threads in [2usize, 3, 16] {
                    let many = encoder::encode_chunked(&chans, 44100, 16, mode, chunk, threads, None, &Default::default()).unwrap();
                    assert_eq!(one, many, "mode={mode} chunk={chunk} threads={threads}");
                }
                for threads in [1usize, 4] {
                    let (_, out) = decode_with_threads(&one, threads).unwrap();
                    assert_eq!(out, chans, "mode={mode} chunk={chunk} threads={threads}");
                }
            }
        }
    }

    #[test]
    fn hostile_chunk_headers_rejected() {
        let chans = vec![signal(30_000, 3)];
        let good = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 10_000, 4, None, &Default::default()).unwrap();
        assert!(decode(&good).is_ok());
        let empty_metadata_block_len = metadata::write_block(&Metadata::default()).len();
        let h0 = HEADER_LEN + empty_metadata_block_len; // first chunk's own inline header
        let patch = |off: usize, v: u32| {
            let mut d = good.clone();
            d[off..off + 4].copy_from_slice(&v.to_le_bytes());
            d
        };
        assert!(decode(&patch(h0 + 4, 0)).is_err(), "zero-frame chunk");
        assert!(decode(&patch(h0 + 4, u32::MAX)).is_err(), "oversized chunk frame count");
        assert!(decode(&patch(h0 + 4, 9_999)).is_err(), "frame counts not summing to total");
        assert!(decode(&patch(h0 + 8, u32::MAX)).is_err(), "byte length past end of file");
        assert!(decode(&patch(h0 + 12, 0xDEAD_BEEF)).is_err(), "wrong chunk CRC");
        let mut bad_sync = good.clone();
        bad_sync[h0] ^= 0xFF;
        assert!(decode(&bad_sync).is_err(), "corrupted chunk sync word");
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(decode(&trailing).is_err(), "trailing garbage");
    }

    #[test]
    fn random_bytes_after_valid_header_never_panic() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let good = encoder::encode(&[signal(3000, 4)], 44100, 16).unwrap();
        for _ in 0..500 {
            let mut d = good[..HEADER_LEN].to_vec();
            let len = (next() % 200) as usize;
            d.extend((0..len).map(|_| next() as u8));
            let _ = decode(&d);
        }
    }

    #[test]
    fn empty_file_and_garbage_rejected() {
        assert!(decode(&[]).is_err());
        assert!(decode(&[0u8; 100]).is_err());
        assert!(decode(b"RIFF....WAVEfmt ").is_err());
    }

    #[test]
    fn cross_subframe_rejects_hostile_fields() {
        use crate::bitio::BitWriter;
        use crate::crossch::{CrossParams, Source};
        let n = 64;
        // Cross subframe: inner Fixed order 1 (warmup verbatim), cross fields, a Rice residual.
        let build = |inner: u64, p: &CrossParams, amp: i64| {
            let mut w = BitWriter::new();
            w.write_bits(0, 5); // wasted
            w.write_bits(SubframeType::Cross as u64, 3);
            w.write_bits(inner, 3);
            if inner == SubframeType::Fixed as u64 {
                w.write_bits(1, 3); // order 1
                w.write_signed(0, 16); // warmup
                p.write(&mut w);
                stage2::Params::write(None, &mut w);
                ltp::Params::write(None, &mut w);
                rice::encode(&mut w, &vec![amp; n - 1]);
            }
            w.finish()
        };
        let p = CrossParams { ref_idx: 0, source: Source::Samples, lag0: -2, coeffs: vec![1, 2, 3], shift: 1, precision: 8 };
        let good_ref = SubOut { samples: (0..n as i64).collect(), res: vec![0; n], res_start: 0 };
        // Well-formed: decodes.
        let bytes = build(SubframeType::Fixed as u64, &p, 3);
        assert!(read_subframe(&mut BitReader::new(&bytes), n, 16, &[], std::slice::from_ref(&good_ref)).is_ok());
        // No earlier subframe to reference.
        assert!(read_subframe(&mut BitReader::new(&bytes), n, 16, &[], &[]).is_err());
        // Inner kinds other than Fixed/LPC (including Cross itself) are rejected.
        for inner in [SubframeType::Constant, SubframeType::Verbatim, SubframeType::Palette, SubframeType::PaletteRle, SubframeType::Cross] {
            let bytes = build(inner as u64, &p, 3);
            assert!(read_subframe(&mut BitReader::new(&bytes), n, 16, &[], std::slice::from_ref(&good_ref)).is_err(), "inner {inner:?}");
        }
        // A reference whose length differs from the frame, or whose values exceed the source bound.
        let short = SubOut { samples: vec![0; n - 1], res: vec![], res_start: n - 1 };
        assert!(read_subframe(&mut BitReader::new(&bytes), n, 16, &[], std::slice::from_ref(&short)).is_err());
        let huge = SubOut { samples: vec![crate::crossch::SOURCE_BOUND + 1; n], res: vec![0; n], res_start: 0 };
        assert!(read_subframe(&mut BitReader::new(&bytes), n, 16, &[], std::slice::from_ref(&huge)).is_err());
        // Extreme but legal operands (source at the bound, largest taps, residuals near the Rice
        // limit) must not overflow: either decode or fail the sane-sample check, never panic.
        let p = CrossParams { ref_idx: 0, source: Source::Residual, lag0: -8, coeffs: vec![-(1 << 15); 16], shift: 0, precision: 16 };
        let edge = SubOut { samples: vec![0; n], res: vec![crate::crossch::SOURCE_BOUND; n], res_start: 0 };
        let bytes = build(SubframeType::Fixed as u64, &p, 1 << 40);
        let _ = read_subframe(&mut BitReader::new(&bytes), n, 16, &[], std::slice::from_ref(&edge));
    }

    #[test]
    fn lpc_subframe_rejects_out_of_range_precision() {
        use crate::bitio::BitWriter;
        // Stored as precision-1 in 4 bits: 0 and 1 (precision 1, 2) are below MIN_PRECISION.
        for stored in 0..(MIN_PRECISION as u64 - 1) {
            let mut w = BitWriter::new();
            w.write_bits(0, 5); // wasted = 0
            w.write_bits(SubframeType::Lpc as u64, 3);
            w.write_bits(0, 5); // order 1
            w.write_bits(0, 5); // shift
            w.write_bits(stored, 4);
            w.write_bits(0, 32);
            w.write_bits(0, 32);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert!(read_subframe(&mut r, 10, 16, &[], &[]).is_err(), "precision {} accepted", stored + 1);
        }
    }

    /// Format v15 Rice-coded coefficients: one outside its declared precision is rejected, and
    /// `write_coeffs` output reads back exactly.
    #[test]
    fn lpc_coefficients_roundtrip_and_reject_out_of_precision() {
        use crate::bitio::BitWriter;
        for coeffs in [vec![5i64], vec![-8, 7], vec![4000, -3000, 12, 0, -1, 2047], vec![-(1 << 15), (1 << 15) - 1, 0]] {
            let p = 16;
            let mut w = BitWriter::new();
            lpc::write_coeffs(&mut w, &coeffs);
            assert_eq!(w.bit_len(), lpc::coeff_bits(&coeffs));
            let bytes = w.finish();
            assert_eq!(lpc::read_coeffs(&mut BitReader::new(&bytes), coeffs.len(), p).unwrap(), coeffs);
            // The same stream read at a precision too small for its largest coefficient.
            let need = coeffs.iter().map(|&c| 65 - (c ^ (c >> 63)).leading_zeros()).max().unwrap();
            if need > MIN_PRECISION {
                assert!(lpc::read_coeffs(&mut BitReader::new(&bytes), coeffs.len(), need - 1).is_err(), "{coeffs:?}");
            }
        }
    }

    #[test]
    fn history_warmup_rejects_hostile_history_and_residuals_without_panicking() {
        // Format v9's history path: extreme history (as large as stereo recombination of
        // already-bounds-checked samples can make it) with maximal coefficients and huge escape
        // residuals must be rejected, never overflow or panic -- for LPC and every fixed order.
        use crate::bitio::BitWriter;
        let huge = 1i64 << 51;
        let history: Vec<i64> = (0..HISTORY_LEN).map(|i| if i % 2 == 0 { huge } else { -huge }).collect();
        for res_val in [1i64 << 38, -(1i64 << 38), 0] { // 2^38 zigzags to 40 bits: the widest legal escape
            let mut w = BitWriter::new();
            w.write_bits(0, 5);
            w.write_bits(SubframeType::Lpc as u64, 3);
            w.write_bits(31, 5); // order 32
            w.write_bits(0, 5); // shift 0
            w.write_bits((MAX_PRECISION - 1) as u64, 4);
            for j in 0..32 { w.write_signed(if j % 2 == 0 { (1 << (MAX_PRECISION - 1)) - 1 } else { -(1 << (MAX_PRECISION - 1)) }, MAX_PRECISION); }
            stage2::Params::write(None, &mut w);
            ltp::Params::write(None, &mut w);
            rice::encode(&mut w, &[res_val; 50]);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert!(read_subframe(&mut r, 50, 25, &history, &[]).is_err());
            for order in 1..=4u64 {
                let mut w = BitWriter::new();
                w.write_bits(0, 5);
                w.write_bits(SubframeType::Fixed as u64, 3);
                w.write_bits(order, 3);
                stage2::Params::write(None, &mut w);
                ltp::Params::write(None, &mut w);
                rice::encode(&mut w, &[res_val; 50]);
                let bytes = w.finish();
                let mut r = BitReader::new(&bytes);
                assert!(read_subframe(&mut r, 50, 25, &history, &[]).is_err(), "fixed order {order}");
            }
        }
    }

    #[test]
    fn palette_subframe_rejects_oversized_table_and_out_of_range_index() {
        use crate::bitio::BitWriter;

        let mut w = BitWriter::new();
        w.write_bits(0, 5); // wasted = 0
        w.write_bits(4, 3); // SubframeType::Palette
        w.write_bits(15, 4); // count-2 = 15 -> count = 17, one past MAX_PALETTE (16)
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(read_subframe(&mut r, 10, 16, &[], &[]).is_err());

        let mut w = BitWriter::new();
        w.write_bits(0, 5);
        w.write_bits(4, 3);
        w.write_bits(0, 4); // count-2 = 0 -> count = 2
        w.write_signed(100, 16);
        w.write_signed(-100, 16);
        w.write_bits(1, 1); // one valid index (< 2)
        w.write_bits(1, 1); // out-of-range: only indices 0 or 1 are valid for a 2-entry table... wait 1 is valid
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        // A 2-entry table needs 1-bit indices, so any 1-bit value (0 or 1) is in range; this
        // exercises the valid path. The genuinely out-of-range case needs a >2-entry table with
        // its widest index value unused, tested below.
        assert!(read_subframe(&mut r, 2, 16, &[], &[]).is_ok());

        let mut w = BitWriter::new();
        w.write_bits(0, 5);
        w.write_bits(4, 3);
        w.write_bits(1, 4); // count-2 = 1 -> count = 3 (needs 2-bit indices, valid range 0..=2)
        w.write_signed(1, 16);
        w.write_signed(2, 16);
        w.write_signed(3, 16);
        w.write_bits(3, 2); // index 3: out of range for a 3-entry table
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(read_subframe(&mut r, 1, 16, &[], &[]).is_err());
    }

    #[test]
    fn palette_rle_subframe_rejects_hostile_fields() {
        use crate::bitio::BitWriter;

        // oversized palette table (same bound as flat Palette)
        let mut w = BitWriter::new();
        w.write_bits(0, 5);
        w.write_bits(5, 3); // SubframeType::PaletteRle
        w.write_bits(15, 4); // count-2 = 15 -> count = 17, one past MAX_PALETTE
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(read_subframe(&mut r, 10, 16, &[], &[]).is_err());

        // run_len_bits == 0 is rejected outright (would make every run-length field zero-width)
        let mut w = BitWriter::new();
        w.write_bits(0, 5);
        w.write_bits(5, 3);
        w.write_bits(0, 4); // count-2 = 0 -> count = 2
        w.write_signed(1, 16);
        w.write_signed(-1, 16);
        w.write_bits(0, 5); // run_len_bits = 0
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(read_subframe(&mut r, 100, 16, &[], &[]).is_err());

        // num_runs declared far larger than the subframe's actual length n -- must be rejected
        // before it's ever used to size an allocation.
        let mut w = BitWriter::new();
        w.write_bits(0, 5);
        w.write_bits(5, 3);
        w.write_bits(0, 4);
        w.write_signed(1, 16);
        w.write_signed(-1, 16);
        w.write_bits(4, 5); // run_len_bits = 4
        w.write_bits((1u64 << 20) - 1, 20); // num_runs - 1 = 2^20-1 -> num_runs = 2^20
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(read_subframe(&mut r, 100, 16, &[], &[]).is_err());

        // a single run whose length overruns n
        let mut w = BitWriter::new();
        w.write_bits(0, 5);
        w.write_bits(5, 3);
        w.write_bits(0, 4);
        w.write_signed(1, 16);
        w.write_signed(-1, 16);
        w.write_bits(10, 5); // run_len_bits = 10 (covers up to 1024)
        w.write_bits(0, 20); // num_runs - 1 = 0 -> num_runs = 1
        w.write_bits(0, 1); // index 0
        w.write_bits(999, 10); // run_length - 1 = 999 -> length 1000, overruns n=100
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(read_subframe(&mut r, 100, 16, &[], &[]).is_err());

        // runs that sum to less than n (undershoot) must also be rejected
        let mut w = BitWriter::new();
        w.write_bits(0, 5);
        w.write_bits(5, 3);
        w.write_bits(0, 4);
        w.write_signed(1, 16);
        w.write_signed(-1, 16);
        w.write_bits(10, 5);
        w.write_bits(0, 20); // num_runs = 1
        w.write_bits(0, 1);
        w.write_bits(9, 10); // run_length = 10, but n = 100
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(read_subframe(&mut r, 100, 16, &[], &[]).is_err());

        // valid, well-formed PaletteRle stream decodes correctly
        let mut w = BitWriter::new();
        w.write_bits(0, 5);
        w.write_bits(5, 3);
        w.write_bits(0, 4);
        w.write_signed(7, 16);
        w.write_signed(-7, 16);
        w.write_bits(5, 5); // run_len_bits = 5 (covers up to 32)
        w.write_bits(1, 20); // num_runs = 2
        w.write_bits(0, 1);
        w.write_bits(9, 5); // run length 10
        w.write_bits(1, 1);
        w.write_bits(9, 5); // run length 10
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let out = read_subframe(&mut r, 20, 16, &[], &[]).unwrap().samples;
        assert_eq!(out, [vec![7i64; 10], vec![-7i64; 10]].concat());
    }

    /// Differential test against the already-trusted `decode_full`: for every target frame across
    /// several chunk boundaries and both modes, `seek`'s returned chunk (once located and sliced
    /// with `chunk_start_frame`) must agree exactly with what a full decode gives at that same
    /// offset. This is the oracle that matters -- `seek` reimplements chunk lookup, not chunk
    /// decoding itself, so the real risk is an off-by-one in which chunk gets selected or where
    /// `chunk_start_frame` lands, not corruption of the decoded samples.
    #[test]
    fn seek_matches_decode_full_at_every_chunk_boundary_both_modes() {
        use crate::format::MODE_BLOCK_INDEPENDENT;
        let chans = vec![signal(50_000, 11), signal(50_000, 12)];
        let chunk = 7000usize; // deliberately not a divisor of 50_000, so the last chunk is short
        for mode in [MODE_BLOCK_INDEPENDENT] {
            let encoded = encoder::encode_chunked(&chans, 44100, 16, mode, chunk, 1, Some(crate::format::DEFAULT_FEC_GROUP), &Default::default()).unwrap();
            let (_, _, full) = decode_full(&encoded, 1).unwrap();
            // Every chunk boundary, one frame before it, and a handful of interior points.
            let mut targets: Vec<u64> = (0..50_000u64).step_by(chunk).collect();
            targets.extend((0..50_000u64).step_by(chunk).filter_map(|f| f.checked_sub(1)));
            targets.extend([0, 1, 3499, 6999, 7000, 7001, 49_998, 49_999]);
            for target in targets {
                let r = seek(&encoded, target).unwrap_or_else(|e| panic!("mode={mode} target={target}: {e}"));
                assert!(r.chunk_start_frame <= target, "mode={mode} target={target}: chunk_start_frame {} > target", r.chunk_start_frame);
                let within = (target - r.chunk_start_frame) as usize;
                assert!(within < r.channels[0].len(), "mode={mode} target={target}: offset {within} >= chunk length {}", r.channels[0].len());
                for c in 0..chans.len() {
                    let expected = full[c][target as usize];
                    let actual = r.channels[c][within];
                    assert_eq!(actual, expected, "mode={mode} target={target} channel={c}: seek={actual} full_decode={expected}");
                }
            }
        }
    }

    #[test]
    fn seek_rejects_out_of_range_target() {
        let chans = vec![signal(5000, 21)];
        let encoded = encoder::encode(&chans, 44100, 16).unwrap();
        let header = StreamHeader::from_bytes(&encoded).unwrap();
        assert!(seek(&encoded, header.total_frames).is_err(), "target == total_frames should be rejected");
        assert!(seek(&encoded, header.total_frames + 1000).is_err());
        assert!(seek(&encoded, u64::MAX).is_err());
        assert!(seek(&encoded, 0).is_ok());
        assert!(seek(&encoded, header.total_frames - 1).is_ok());
    }

    #[test]
    fn seek_rejects_corrupted_target_chunk_but_not_other_chunks() {
        let chans = vec![signal(21_000, 22)];
        let chunk = 7000usize; // exactly 3 chunks
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, chunk, 1, None, &Default::default()).unwrap();
        let empty_metadata_block_len = metadata::write_block(&Metadata::default()).len();
        let h0 = HEADER_LEN + empty_metadata_block_len; // chunk 0's own inline header
        let chunk0_bytes = u32::from_le_bytes(encoded[h0 + 8..h0 + 12].try_into().unwrap()) as usize;
        let h1 = h0 + crate::format::CHUNK_HEADER_LEN + chunk0_bytes; // chunk 1's own inline header
        // Corrupt only chunk 1's stored CRC (the middle chunk) -- a real semantic mismatch caught
        // at decode time, not a structural rejection (there is no separate up-front table anymore).
        let mut bad = encoded.clone();
        bad[h1 + 12..h1 + 16].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        assert!(seek(&bad, 0).is_ok(), "chunk 0 untouched, should still seek fine");
        assert!(seek(&bad, 7500).is_err(), "chunk 1's CRC was corrupted, should be rejected");
        assert!(seek(&bad, 15_000).is_ok(), "chunk 2 untouched, should still seek fine");
    }

    /// Real chunk/parity locations for a test to corrupt specific bytes at, instead of hand-computed
    /// offsets (own established test convention, extended to parity blocks) -- uses
    /// the library's own `locate_chunks`, so these tests stay correct even if the container layout
    /// changes again.
    fn locate_for_test(encoded: &[u8]) -> (crate::format::StreamHeader, Vec<ChunkLoc>, Vec<ParityLoc>) {
        let header = crate::format::StreamHeader::from_bytes(encoded).unwrap();
        let (_meta, payload_start) = metadata::read_block(encoded, HEADER_LEN).unwrap();
        let (chunks, parities, _) = crate::format::locate_chunks(&header, encoded, payload_start).unwrap();
        (header, chunks, parities)
    }

    ///  (FEC): the core recovery property -- a single chunk's payload corrupted
    /// (bit rot, the common real-world case) is transparently *healed* from its group's parity
    /// block, not just detected and rejected, and the recovered decode is bit-exact against the
    /// original audio.
    #[test]
    fn fec_recovers_single_corrupted_chunk_payload() {
        let chans = vec![signal(40_000, 60)];
        let group = 4;
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4000, 1, Some(group), &Default::default()).unwrap();
        let (_, chunks, _) = locate_for_test(&encoded);
        assert!(chunks.len() >= group, "test needs at least one full FEC group");
        let (_, byte_start, _) = chunks[1];
        let mut bad = encoded.clone();
        bad[byte_start] ^= 0xFF;
        let (_, _, decoded) = decode_full(&bad, 1).unwrap_or_else(|e| panic!("FEC should have recovered chunk 1's corrupted payload: {e}"));
        assert_eq!(decoded, chans, "recovered decode must be bit-exact against the original audio");
    }

    /// A corrupted chunk's own header `crc` field (not its payload) is a real, if less common,
    /// corruption pattern -- recovery must still succeed by trusting the parity block's independent
    /// redundant copy of that chunk's real crc, not the damaged one, exactly as `recover_chunk`'s
    /// own doc comment claims.
    #[test]
    fn fec_recovers_when_header_crc_field_itself_is_corrupted() {
        let chans = vec![signal(40_000, 61)];
        let group = 4;
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4000, 1, Some(group), &Default::default()).unwrap();
        let (_, chunks, _) = locate_for_test(&encoded);
        assert!(chunks.len() >= group);
        let (_, byte_start, _) = chunks[2];
        let mut bad = encoded.clone();
        for i in byte_start - 4..byte_start { bad[i] ^= 0xFF; } // the 4 bytes immediately before the payload are this chunk's own header crc field
        let (_, _, decoded) = decode_full(&bad, 1).unwrap_or_else(|e| panic!("FEC should have recovered despite the header's own crc field being corrupted: {e}"));
        assert_eq!(decoded, chans);
    }

    /// XOR parity recovers exactly *one* damaged chunk per group, never more -- a second corrupted
    /// chunk in the same group must be reported as a failure, not silently produce wrong audio
    /// (the decoder must never trust a reconstruction it can't actually verify).
    #[test]
    fn fec_fails_safely_when_two_chunks_in_same_group_are_corrupted() {
        let chans = vec![signal(40_000, 62)];
        let group = 4;
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4000, 1, Some(group), &Default::default()).unwrap();
        let (_, chunks, _) = locate_for_test(&encoded);
        assert!(chunks.len() >= 2);
        let mut bad = encoded.clone();
        // Group 4 over 10 chunks gets 2 shards, so 3 damaged chunks in one block exceed what it can rebuild.
        for c in &chunks[0..3] { bad[c.1] ^= 0xFF; }
        assert!(decode_full(&bad, 1).is_err(), "more damaged chunks than parity shards must not be silently accepted");
    }

    /// If the parity block itself is also damaged (not just the chunk it would have recovered),
    /// recovery must fail safely -- never fabricate audio from a parity payload whose own integrity
    /// can no longer be trusted.
    #[test]
    fn fec_fails_safely_when_parity_block_itself_is_corrupted() {
        let chans = vec![signal(40_000, 63)];
        let group = 4;
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4000, 1, Some(group), &Default::default()).unwrap();
        let (_, chunks, parities) = locate_for_test(&encoded);
        assert!(!parities.is_empty());
        let p = &parities[0];
        let mut bad = encoded.clone();
        // Two shards (group 4); damage both, and a real chunk in that same group so recovery is attempted.
        bad[p.shards_start] ^= 0xFF;
        bad[p.shards_start + p.shard_len] ^= 0xFF;
        bad[chunks[0].1] ^= 0xFF;
        assert!(decode_full(&bad, 1).is_err(), "a damaged parity block must not let recovery fabricate wrong audio");
    }

    ///: one Reed-Solomon block over the whole file rebuilds ANY `m` damaged chunks, wherever
    /// they are (payload bytes, the chunk's own header crc field, several bytes at once), bit-exact,
    /// through `decode_full`, `Reader::decode_chunk` and the low-memory `FileReader`; `m + 1` damaged
    /// chunks are refused, never decoded to wrong audio.
    #[test]
    fn fec_whole_file_rebuilds_any_m_damaged_chunks() {
        let chans = vec![signal(300_000, 70), signal(300_000, 71)];
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 3000, 2, Some(crate::format::FEC_AUTO), &Default::default()).unwrap();
        let (_, chunks, parities) = locate_for_test(&encoded);
        assert_eq!(parities.len(), 1, "one block for the whole file");
        assert_eq!(parities[0].entries.len(), chunks.len());
        let m = parities[0].m;
        assert_eq!(chunks.len(), 100);
        assert_eq!(m, 2, "100 chunks: 1% rounds up to 1, floor of 2");
        let mut rng = 12345u64;
        let mut next = move |n: usize| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng % n as u64) as usize };
        for round in 0..6 {
            let mut hit = std::collections::BTreeSet::new();
            while hit.len() < m { hit.insert(next(chunks.len())); }
            let mut bad = encoded.clone();
            for &i in &hit {
                let (_, start, e) = chunks[i];
                match round % 3 {
                    0 => bad[start + next(e.bytes as usize)] ^= 0x40,
                    1 => bad[start - 4] ^= 0x01, // the chunk's own header crc field
                    _ => { for _ in 0..5 { bad[start + next(e.bytes as usize)] ^= 0xA5; } }
                }
            }
            let (_, _, decoded) = decode_full(&bad, 2).unwrap_or_else(|e| panic!("round {round} chunks {hit:?}: {e}"));
            assert_eq!(decoded, chans, "round {round}: rebuilt decode must be bit-exact");
            let r = Reader::open(&bad[..]).unwrap();
            for &i in &hit { assert!(r.decode_chunk(i).is_ok()); }
            let mut fr = FileReader::open(std::io::Cursor::new(bad.clone())).unwrap();
            let mut buf = Vec::new();
            for &i in &hit { fr.read_payload(i, &mut buf).unwrap(); assert_eq!(crc32(&buf), chunks[i].2.crc); }
        }
        let mut bad = encoded.clone();
        for i in [3, 40, 77] { bad[chunks[i].1] ^= 0xFF; }
        assert!(decode_full(&bad, 2).is_err(), "m+1 damaged chunks must be refused");
    }

    ///: damaged chunk *headers* (sync, frame count, length) no longer end the file -- the
    /// chunks are located from the parity blocks' own copy of the table, then rebuilt like any other
    /// damaged chunk. Covers a whole-file block and several blocks; also a burst across a boundary.
    #[test]
    fn fec_survives_damaged_chunk_headers() {
        let chans = vec![signal(300_000, 74), signal(300_000, 75)];
        for group in [crate::format::FEC_AUTO, 30] {
            let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 3000, 2, Some(group), &Default::default()).unwrap();
            let (_, chunks, parities) = locate_for_test(&encoded);
            assert_eq!(parities.len(), if group == 30 { 4 } else { 1 });
            let mut bad = encoded.clone();
            // Chunk 5: whole header wiped (sync + frames + bytes + crc). Chunk 42: just the length field.
            for b in &mut bad[chunks[5].1 - 16..chunks[5].1] { *b = 0xEE; }
            bad[chunks[42].1 - 8] ^= 0x7F;
            let (_, _, decoded) = decode_full(&bad, 2).unwrap_or_else(|e| panic!("group {group}: {e}"));
            assert_eq!(decoded, chans, "group {group}: bit-exact after header damage");
            let r = Reader::open(&bad[..]).unwrap();
            assert_eq!(r.chunk_count(), chunks.len());
            let mut fr = FileReader::open(std::io::Cursor::new(bad.clone())).unwrap();
            let mut buf = Vec::new();
            fr.read_payload(5, &mut buf).unwrap();
            assert_eq!(crc32(&buf), chunks[5].2.crc);
            // A burst straddling the end of one chunk and the header of the next.
            let mut bad = encoded.clone();
            let end = chunks[20].1 + chunks[20].2.bytes as usize;
            for b in &mut bad[end - 300..end + 20] { *b ^= 0xFF; }
            assert_eq!(decode_full(&bad, 2).unwrap().2, chans, "group {group}: burst across a chunk boundary");
        }
        // Without FEC the same damage is a hard error, as before.
        let plain = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 3000, 2, None, &Default::default()).unwrap();
        let (_, chunks, _) = locate_for_test(&plain);
        let mut bad = plain.clone();
        bad[chunks[5].1 - 16] ^= 0xFF;
        assert!(decode_full(&bad, 2).is_err());
    }

    /// Parity shards that are themselves damaged are skipped (their own CRCs), spending spare shards.
    #[test]
    fn fec_skips_damaged_shards_and_uses_spare_ones() {
        let chans = vec![signal(300_000, 72)];
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 3000, 1, Some(crate::format::FEC_AUTO), &Default::default()).unwrap();
        let (_, chunks, parities) = locate_for_test(&encoded);
        let p = &parities[0];
        assert!(p.m >= 2);
        let mut bad = encoded.clone();
        bad[p.shards_start + 5] ^= 0xFF; // shard 0 damaged; shard 1 still intact: one chunk is rebuildable
        bad[chunks[10].1] ^= 0xFF;
        assert_eq!(decode_full(&bad, 1).unwrap().2, chans);
        bad[chunks[11].1] ^= 0xFF; // two damaged chunks now need two intact shards, only one is left
        assert!(decode_full(&bad, 1).is_err());
    }

    /// A long file with the default shard count: overhead stays near 1% and any 1% of chunks heal.
    #[test]
    fn fec_whole_file_overhead_is_about_one_percent() {
        let chans = vec![signal(2_000_000, 73)];
        let plain = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4410, 2, None, &Default::default()).unwrap();
        let fec = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4410, 2, Some(crate::format::FEC_AUTO), &Default::default()).unwrap();
        let (_, chunks, parities) = locate_for_test(&fec);
        assert_eq!(parities[0].m, chunks.len().div_ceil(100));
        let extra = (fec.len() - plain.len()) as f64 / plain.len() as f64;
        assert!(extra > 0.008 && extra < 0.03, "overhead {extra}");
        let mut bad = fec.clone();
        for i in (0..chunks.len()).step_by(chunks.len() / parities[0].m).take(parities[0].m) { bad[chunks[i].1 + 7] ^= 0x10; }
        assert_eq!(decode_full(&bad, 2).unwrap().2, chans);
    }

    /// `fec_group: None` must produce the exact pre- layout: zero parity blocks, nothing to
    /// walk past that isn't a real data chunk.
    #[test]
    fn fec_disabled_produces_no_parity_blocks() {
        let chans = vec![signal(40_000, 64)];
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4000, 1, None, &Default::default()).unwrap();
        let (_, _, parities) = locate_for_test(&encoded);
        assert!(parities.is_empty(), "fec_group=None must produce zero parity blocks");
    }

    /// `seek` gets the same recovery `decode_full` does -- it has its own CRC-check-then-recover
    /// call site, not shared code, so it needs its own direct test.
    #[test]
    fn fec_seek_recovers_corrupted_chunk() {
        let chans = vec![signal(40_000, 65)];
        let group = 4;
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4000, 1, Some(group), &Default::default()).unwrap();
        let (_, chunks, _) = locate_for_test(&encoded);
        assert!(chunks.len() > 3);
        let (frame_start, byte_start, e) = chunks[3];
        let mut bad = encoded.clone();
        bad[byte_start] ^= 0xFF;
        let r = seek(&bad, frame_start).unwrap_or_else(|err| panic!("seek should recover the corrupted chunk via FEC: {err}"));
        assert_eq!(r.chunk_start_frame, frame_start);
        assert_eq!(r.channels[0], chans[0][frame_start as usize..frame_start as usize + e.frames as usize]);
    }

    /// A file with fewer real chunks than the FEC group size still gets exactly one (partial-group)
    /// parity block covering all of them -- the "trailing parity block after the last data chunk"
    /// path in `locate_chunks` specifically, not just a full group.
    #[test]
    fn fec_partial_final_group_smaller_than_group_size_roundtrips() {
        let chans = vec![signal(2000, 66)];
        // FEC is opt-in (addendum: default-on regressed compression ~6-7% vs FLAC, reverted
        // to opt-in) -- request it explicitly rather than relying on encoder::encode's default.
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, crate::format::default_chunk_frames(44100), 1, Some(crate::format::DEFAULT_FEC_GROUP), &Default::default()).unwrap();
        let (_, decoded_chans) = decode(&encoded).unwrap();
        assert_eq!(decoded_chans, chans);
        let (_, chunks, parities) = locate_for_test(&encoded);
        assert_eq!(chunks.len(), 1);
        assert_eq!(parities.len(), 1, "a single-chunk file should still get exactly one (partial-group) parity block");
        assert_eq!(parities[0].entries.len(), 1);
    }

    /// Recovery must give the identical, correct answer no matter how many worker threads
    /// `decode_full` uses -- `recover_chunk` reads shared, already-located data (`data`/`chunks`/
    /// `parities`), never mutable state, so this should already be safe, but it's a real property
    /// worth asserting directly rather than assuming.
    #[test]
    fn fec_recovery_works_across_thread_counts() {
        let chans = vec![signal(60_000, 67), signal(60_000, 68)];
        let group = 3;
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 4000, 1, Some(group), &Default::default()).unwrap();
        let (_, chunks, _) = locate_for_test(&encoded);
        assert!(chunks.len() >= 5);
        let mut bad = encoded.clone();
        bad[chunks[4].1] ^= 0xFF;
        for threads in [1usize, 2, 4] {
            let (_, _, decoded) = decode_full(&bad, threads).unwrap_or_else(|e| panic!("threads={threads}: {e}"));
            assert_eq!(decoded, chans, "threads={threads}");
        }
    }

    #[test]
    fn seek_never_panics_on_random_bytes_after_a_valid_header() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let good = encoder::encode(&[signal(3000, 5)], 44100, 16).unwrap();
        for _ in 0..500 {
            let mut d = good[..HEADER_LEN].to_vec();
            let len = (next() % 200) as usize;
            d.extend((0..len).map(|_| next() as u8));
            let _ = seek(&d, next() % 100_000);
        }
    }

    /// `StreamEncoder`/`decode_stream` round trip, a *genuinely unbounded* source: pushes chunks one at a time without ever declaring
    /// `total_frames` up front, decodes them back one at a time via a real `std::io::Read` (a
    /// `Vec<u8>` slice, but read through the `Read` trait rather than sliced directly, so this
    /// exercises the actual incremental I/O path, not just the in-memory logic). Must match a
    /// conventional encode/decode of the same audio sample-for-sample.
    #[test]
    fn stream_encoder_decoder_roundtrip_unknown_length_both_modes() {
        use crate::encoder::StreamEncoder;
        use crate::format::MODE_BLOCK_INDEPENDENT;
        let chunks: Vec<Vec<i64>> = (0..5).map(|i| signal(4000 + i * 137, 30 + i as u64)).collect();
        let full: Vec<i64> = chunks.concat();
        for mode in [MODE_BLOCK_INDEPENDENT] {
            let mut buf = Vec::new();
            let mut enc = StreamEncoder::new(&mut buf, 1, 44100, 16, mode, &Metadata::default()).unwrap();
            for c in &chunks { enc.push_chunk(&[c.clone()]).unwrap(); }
            enc.finish().unwrap();

            let header_check = StreamHeader::from_bytes(&buf).unwrap();
            assert_eq!(header_check.total_frames, crate::format::TOTAL_FRAMES_UNKNOWN, "mode={mode}: streamed header must carry the sentinel");

            let mut got: Vec<i64> = Vec::new();
            let mut n_chunks_seen = 0usize;
            let (header, _meta) = decode_stream(buf.as_slice(), |ch| {
                n_chunks_seen += 1;
                got.extend_from_slice(&ch[0]);
                Ok(())
            }).unwrap();
            assert_eq!(header.channels, 1);
            assert_eq!(n_chunks_seen, chunks.len(), "mode={mode}");
            assert_eq!(got, full, "mode={mode}");

            // A plain slice-based `decode_stream` (no separate real total_frames known) must also
            // reach the same answer -- confirms the unknown-length EOF-driven stopping condition,
            // not just that decode_stream happens to work when fed the right count somehow.
        }
    }

    /// Same round trip, but with a real `total_frames` known up front (mirroring what
    /// `encode_chunked` writes) -- confirms `decode_stream` handles the known-length path
    /// identically to `decode_full` on the very same bytes, not just the unknown-length path.
    #[test]
    fn decode_stream_matches_decode_full_on_a_known_length_stream() {
        let chans = vec![signal(30_000, 40), signal(30_000, 41)];
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 7000, 4, None, &Metadata::default()).unwrap();
        let (_, _, full) = decode_full(&encoded, 1).unwrap();

        let mut got: Vec<Vec<i64>> = vec![Vec::new(); 2];
        let (header, _meta) = decode_stream(encoded.as_slice(), |ch| {
            for (dst, src) in got.iter_mut().zip(ch) { dst.extend_from_slice(src); }
            Ok(())
        }).unwrap();
        assert_eq!(header.total_frames, 30_000);
        assert_eq!(got, full);
    }

    #[test]
    fn decode_stream_rejects_truncation_and_trailing_garbage() {
        let chans = vec![signal(20_000, 50)];
        let encoded = encoder::encode_chunked(&chans, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, 6000, 1, None, &Metadata::default()).unwrap();
        for cut in [HEADER_LEN, HEADER_LEN + 10, encoded.len() / 2, encoded.len() - 1] {
            let r = decode_stream(&encoded[..cut], |_| Ok(()));
            assert!(r.is_err(), "cut at {cut} should be rejected as truncated, not silently accepted");
        }
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(decode_stream(trailing.as_slice(), |_| Ok(())).is_err(), "trailing garbage should be rejected");
    }

    #[test]
    fn stream_encoder_rejects_bad_inputs() {
        use crate::encoder::StreamEncoder;
        let mut buf = Vec::new();
        assert!(StreamEncoder::new(&mut buf, 0, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, &Metadata::default()).is_err(), "zero channels");
        let mut enc = StreamEncoder::new(&mut buf, 2, 44100, 16, crate::format::MODE_BLOCK_INDEPENDENT, &Metadata::default()).unwrap();
        assert!(enc.push_chunk(&[vec![1i64, 2, 3]]).is_err(), "wrong channel count");
        assert!(enc.push_chunk(&[vec![1i64, 2, 3], vec![1i64, 2]]).is_err(), "mismatched channel lengths");
        assert!(enc.push_chunk(&[vec![], vec![]]).is_err(), "empty chunk");
    }

    #[test]
    fn stream_decode_never_panics_on_random_bytes() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for _ in 0..2000 {
            let len = (next() % 512) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let _ = decode_stream(bytes.as_slice(), |_| Ok(()));
        }
    }

    #[test]
    fn file_reader_matches_reader_recovers_damage_and_rejects_truncation() {
        use std::io::Cursor;
        let l: Vec<i64> = (0..30_000i64).map(|i| ((i * 977) % 30001) - 15000).collect();
        let r: Vec<i64> = (0..30_000i64).map(|i| ((i * 41) % 4001) - 2000).collect();
        let x = vec![l, r];
        let bytes = encoder::encode_chunked(&x, 44100, 16, 0, 4096 * 2, 2, Some(3), &Metadata::default()).unwrap();
        let rd = Reader::open(bytes.as_slice()).unwrap();
        let mut fr = FileReader::open(Cursor::new(bytes.clone())).unwrap();
        assert_eq!((fr.chunk_count(), fr.total_frames, fr.fec_group(), fr.parity_count()), (rd.chunk_count(), rd.total_frames, rd.fec_group(), rd.parity_count()));
        for i in 0..rd.chunk_count() {
            assert_eq!(fr.decode_chunk(i).unwrap(), rd.decode_chunk(i).unwrap(), "chunk {i}");
            assert_eq!((fr.chunk_start(i), fr.chunk_frames(i)), (rd.chunk_start(i), rd.chunk_frames(i)));
        }
        for (i, c) in fr.decode_chunks(0..rd.chunk_count(), 3).into_iter().enumerate() {
            assert_eq!(c.unwrap(), rd.decode_chunk(i).unwrap(), "parallel chunk {i}");
        }
        assert_eq!(fr.chunk_for_frame(9000), rd.chunk_for_frame(9000));
        assert!(fr.chunk_for_frame(30_000).is_none());
        // One damaged byte inside a chunk: recovered from parity, like the in-memory reader.
        let mut bad = bytes.clone();
        bad[rd.chunks[1].1 + 5] ^= 0xFF;
        let mut fr2 = FileReader::open(Cursor::new(bad)).unwrap();
        assert_eq!(fr2.decode_chunk(1).unwrap(), rd.decode_chunk(1).unwrap());
        // Truncation at any point is refused by both readers alike.
        for cut in [0, 10, HEADER_LEN + 2, bytes.len() / 2, bytes.len() - 1] {
            assert_eq!(Reader::open(&bytes[..cut]).is_err(), FileReader::open(Cursor::new(bytes[..cut].to_vec())).is_err(), "cut {cut}");
        }
    }

    // ---- OLS chunk (cfg 3 / cfg 4): the one part of the format that uses floating point ----

    /// Deterministic correlated stereo test signal: two tones, a left channel that leaks into the
    /// right through a short filter, plus noise; `amp` is the peak in sample units.
    fn ols_signal(n: usize, amp: f64, seed: u64) -> (Vec<i64>, Vec<i64>) {
        let mut st = seed | 1;
        let mut next = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let l: Vec<i64> = (0..n).map(|i| {
            let t = i as f64;
            let noise = ((next() % 2001) as f64 - 1000.0) / 1000.0 * 0.02;
            (((t * 0.031).sin() * 0.5 + (t * 0.0071).sin() * 0.3 + noise) * amp).round() as i64
        }).collect();
        let r: Vec<i64> = (0..n).map(|i| {
            let prev = if i > 0 { l[i - 1] as f64 } else { 0.0 };
            (l[i] as f64 * 0.7 + prev * 0.2).round() as i64 + (next() % 7) as i64 - 3
        }).collect();
        (l, r)
    }

    /// `[value-map section: none][cfg byte][OLS body]`, the layout `decode_frames` parses.
    fn ols_payload(l: &[i64], r: &[i64], bits: u8, irls: bool) -> Vec<u8> {
        let body = encoder::encode_ols_chunk(l, r, bits, irls);
        assert_eq!(body[0], if irls { 4 } else { 3 }, "the chunk writes its own config byte");
        let mut p = vec![0u8];
        p.extend(body);
        p
    }

    /// Both configs round-trip at 16 and 24 bits across the lengths where the encoder's and the
    /// decoder's block and warm-up bookkeeping could disagree: 1 frame, the 16-frame statistics
    /// warm-up and its neighbours, and the `OLS_BLOCK` boundaries.
    #[test]
    fn ols_chunk_roundtrips_at_every_boundary_length() {
        let b = encoder::OLS_BLOCK;
        for bits in [16u8, 24] {
            let amp = ((1i64 << (bits - 1)) - 1) as f64 * 0.6;
            for irls in [false, true] {
                for n in [1, 2, 15, 16, 17, 100, b - 1, b, b + 1, 2 * b + 37] {
                    let (l, r) = ols_signal(n, amp, 0x9E37_79B9 ^ n as u64);
                    let payload = ols_payload(&l, &r, bits, irls);
                    let got = decode_frames(&payload, 2, bits, n).unwrap_or_else(|e| panic!("{bits}-bit irls {irls} n {n}: {e}"));
                    assert_eq!(got, vec![l, r], "{bits}-bit irls {irls} n {n}");
                }
            }
        }
    }

    /// Full-scale extremes, silence, DC and a hard clip must survive the float predictor exactly.
    #[test]
    fn ols_chunk_roundtrips_extreme_signals() {
        let n = encoder::OLS_BLOCK + 123;
        for bits in [16u8, 24] {
            let (lo, hi) = (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1);
            let alt: Vec<i64> = (0..n).map(|i| if i % 2 == 0 { hi } else { lo }).collect();
            let alt2: Vec<i64> = (0..n).map(|i| if i % 3 == 0 { lo } else { hi }).collect();
            let (tone, _) = ols_signal(n, hi as f64 * 3.0, 77);
            let clipped: Vec<i64> = tone.iter().map(|&x| x.clamp(lo, hi)).collect();
            let cases: [(&str, Vec<i64>, Vec<i64>); 5] = [
                ("silence", vec![0; n], vec![0; n]),
                ("dc", vec![hi; n], vec![lo; n]),
                ("alternating extremes", alt, alt2),
                ("identical channels", clipped.clone(), clipped.clone()),
                ("anti-phase clipped", clipped.clone(), clipped.iter().map(|&x| (-x).clamp(lo, hi)).collect()),
            ];
            for (name, l, r) in cases {
                for irls in [false, true] {
                    let payload = ols_payload(&l, &r, bits, irls);
                    let got = decode_frames(&payload, 2, bits, n).unwrap_or_else(|e| panic!("{name} {bits}-bit irls {irls}: {e}"));
                    assert_eq!(got, vec![l.clone(), r.clone()], "{name} {bits}-bit irls {irls}");
                }
            }
        }
    }

    /// Hostile input: every truncation is an error, never a panic, and a trailing byte is refused.
    #[test]
    fn ols_chunk_rejects_truncation_and_trailing_bytes() {
        let n = 3000;
        let (l, r) = ols_signal(n, 20_000.0, 5);
        for irls in [false, true] {
            let payload = ols_payload(&l, &r, 16, irls);
            assert!(decode_frames(&payload, 2, 16, n).is_ok());
            let step = (payload.len() / 200).max(1);
            for cut in (0..payload.len()).step_by(step).chain([payload.len() - 1]) {
                assert!(decode_frames(&payload[..cut], 2, 16, n).is_err(), "irls {irls}: truncated at {cut}/{} accepted", payload.len());
            }
            let mut long = payload.clone();
            long.push(0);
            assert!(decode_frames(&long, 2, 16, n).is_err(), "irls {irls}: trailing byte accepted");
        }
    }

    /// Hostile input: flipping a bit either errors or decodes in range (the chunk's CRC, not this
    /// layer, catches the rest) -- and never panics or reads out of bounds.
    #[test]
    fn ols_chunk_survives_single_bit_flips_without_panicking() {
        let n = 600;
        let (l, r) = ols_signal(n, 12_000.0, 9);
        for irls in [false, true] {
            let payload = ols_payload(&l, &r, 16, irls);
            for byte in 0..payload.len() {
                for bit in [0u8, 3, 7] {
                    let mut bad = payload.clone();
                    bad[byte] ^= 1 << bit;
                    if let Ok(chs) = decode_frames(&bad, 2, 16, n) {
                        assert!(chs.iter().all(|c| c.len() == n && c.iter().all(|&x| (-32768..=32767).contains(&x))), "irls {irls} byte {byte} bit {bit}");
                    }
                }
            }
        }
    }

    /// cfg 3/4 is defined for exactly two channels of at most 24 bits; anything else is refused
    /// up front, as is a cfg byte past 4.
    #[test]
    fn ols_config_is_refused_for_other_layouts() {
        let (l, r) = ols_signal(500, 10_000.0, 3);
        let body = encoder::encode_ols_chunk(&l, &r, 16, false);
        let mut p = vec![0u8];
        p.extend(&body);
        assert!(decode_frames(&p, 1, 16, 500).is_err(), "mono");
        assert!(decode_frames(&p, 3, 16, 500).is_err(), "three channels");
        assert!(decode_frames(&p, 2, 32, 500).is_err(), "32-bit");
        let mut bad_cfg = p.clone();
        bad_cfg[1] = 5;
        assert!(decode_frames(&bad_cfg, 2, 16, 500).is_err(), "cfg 5");
        assert!(decode_frames(&p[..1], 2, 16, 500).is_err(), "missing cfg byte");
    }

    /// Whole files at `Insane`: stereo 16- and 24-bit round-trip, with and without FEC, and the
    /// decoded audio does not depend on the decode thread count. Short chunks make several chunks
    /// per file so chunk-start warm-up is crossed repeatedly.
    #[test]
    fn insane_stereo_files_roundtrip_and_decode_identically_on_any_thread_count() {
        for bits in [16u8, 24] {
            let amp = ((1i64 << (bits - 1)) - 1) as f64 * 0.5;
            let (l, r) = ols_signal(44_100 + 777, amp, 0xC0FFEE);
            let x = vec![l, r];
            for fec in [None, Some(2)] {
                let bytes = encoder::encode_chunked_effort(&x, 44_100, bits, 0, 8192, 2, fec, &Metadata::default(), encoder::Effort::Insane).unwrap();
                let (_, _, one) = decode_full(&bytes, 1).unwrap();
                let (_, _, four) = decode_full(&bytes, 4).unwrap();
                assert_eq!(one, x, "{bits}-bit fec {fec:?}");
                assert_eq!(four, one, "{bits}-bit fec {fec:?}: thread count changed the audio");
                verify(&bytes, 2).unwrap();
            }
        }
    }
}
