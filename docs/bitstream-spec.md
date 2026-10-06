# FAK bitstream specification

**Specification version 1.1.0 (format version 21).** FAK ("FLAC Audio Killer") files use the extension `.fak`; the
stream header carries the format version byte `21` (the value this specification describes, see "Stream
header").

This document is the authoritative description of the on-disk format. It is implemented in
`src/format.rs`, `src/metadata.rs`, `src/encoder.rs` and `src/decoder.rs`; nothing about the format should
be inferred from implementation details that are not written here.

## Status and compatibility rules

The format was frozen at version 20 (specification 1.0.0) and then revised to add the carried long
filter and the OLS chunk (version 21, specification 1.1.0; see "Chunk config byte"). From
version 1.1.0 of this specification on:

- A conforming decoder accepts exactly the stream header version byte `21` and rejects every other
  value. Version 20 and earlier development versions of the format (bytes 1-19) were never released and are not
  readable.
- A `.fak` file written by any conforming encoder decodes, with any conforming decoder, to the identical
  PCM on every platform and instruction set. The decoder is integer-exact and deterministic.
- Nothing in this document changes in a way that alters how an existing file is decoded. Corrections
  to wording that do not change the format are allowed. Any change to what a decoder must accept or
  produce is a new format version with a new header byte and a new specification.
- Encoders are free to improve (different choices of predictors, partitioning, parameters, search
  effort); the bytes an encoder produces need not be the same between encoders, machines, or runs.
  Only the decoded audio is specified.
- Reserved fields (the header's `mode` byte, which must be 0) stay reserved. A decoder rejects a value it
  does not define rather than guessing.

## Design choices

- **One stream mode, signaled by the header's `mode` byte, which must be 0
  (`MODE_BLOCK_INDEPENDENT`, described below): frames carry no state from earlier frames.** The byte
  is reserved so a future stream mode needs no header change.
- **Every file is a sequence of independent chunks.** No chunk depends on any other
  chunk's contents or decoder state, so an encoder or decoder may process chunks concurrently and
  must produce identical results regardless of how many it runs at once. Each chunk's own inline
  header also serves as a seek index, and the frames within a chunk are additionally
  self-delimiting -- see `decoder::seek`.
- **The header, every chunk, and every frame carry a CRC** so corruption is *detected*,
  never silently misdecoded. The decoder treats its input as hostile: every
  length field is range-checked against the stream's own declared totals and the bytes actually
  present (or, for the streaming reader, a conservative sanity bound -- see "Streaming") before it
  is trusted for allocation or indexing.

## Stream layout

```
[stream header: 56 bytes] [metadata block] [chunk 0: sync+frames+bytes+crc (16 bytes) + payload] ... [chunk N-1: sync+frames+bytes+crc + payload]
```

FEC parity blocks (sync `FPAR`, see "FEC parity block" below) may appear after data chunks --
interspersed with, not separate from, the chunk sequence above: a reader walks chunk-or-parity boundaries one at a time by sync word, never assuming
every boundary is a data chunk. When FEC is disabled (`fec_group: None`) no `FPAR` blocks are
written at all, and the layout is exactly the plain sequence shown above.

A chunk payload is a value-map section (see "Value map"), a chunk config byte (see "Chunk config
byte"), then one or more frames (see "Frame") -- or, for config 3, the OLS payload -- covering exactly
the chunk's sample-frames.

### Chunk header (16 bytes, immediately before each chunk's payload)

| size | field | notes |
|---|---|---|
| 4 | sync | ASCII `"FCHK"` |
| 4 | frames | u32 LE, sample-frames in this chunk (`0 < frames <= 2^22`) |
| 4 | bytes | u32 LE, this chunk's payload length |
| 4 | crc | u32 LE, CRC-32 of this chunk's payload |

A decoder must reject: a bad sync word, `frames == 0` or `frames > 2^22`, a payload that would run
past the end of the data actually available, and a payload whose CRC-32 doesn't match. For a
known-length stream (`total_frames != TOTAL_FRAMES_UNKNOWN`), frame counts across all chunks must
sum exactly to `total_frames`, with no trailing bytes left over after the last chunk. For a
`TOTAL_FRAMES_UNKNOWN` stream, chunks are simply read until the input itself ends -- there is no
declared total to validate against, so a clean end of input at a chunk boundary is the only valid
stopping point (see "Streaming"). `total_frames == 0` means zero chunks.

Locating a specific chunk (e.g. for `decoder::seek`) costs the same as walking the old up-front
table did: every header is a fixed 16 bytes and every payload is skipped over by byte offset, never
decoded, so the scan is O(chunk count) either way.

### Chunk config byte

One byte after the value-map section selects the chunk's carried-filter mode; values above 3 are
rejected.

| cfg | meaning |
|---|---|
| 0 | none: frames are as described below, with no carried filter |
| 1 | carried long filter, 512 taps, update shift k = 8 |
| 2 | carried long filter, 1024 taps, update shift k = 9 |
| 3 | OLS chunk (below); requires 2 channels and bits_per_sample <= 24 |

**Carried filter (cfg 1, 2).** Each subframe slot (channel index; at least 2 slots) owns one filter
whose state persists across all frames of the chunk and starts at zero at the chunk start: `taps` i16
weights (zero), the last `taps` pre-stage-2 residuals r (zero) and a running average `avg` (zero). In every
Fixed/LPC/Cross subframe of a slot, after the cross-channel fields and before the stage-2 header, one
flag bit follows: 1 = the carried filter is used, then `s_plus_16` (6 bits, s = value - 16, s > 31
rejected) and the stage-2 header is NOT present; 0 = the ordinary stage-2 header follows. The residual
read from the Rice stream (after LTP inverse) is the carried filter's output `e`; the decoder recovers
`r[n] = e[n] + pred[n]` (|r| <= 2^40, else the stream is rejected). With `x = clamp(r, +-2^40)`, input
`in(r) = clamp(s >= 0 ? x >> s : x << -s, i16)`, `dot = sum(w[i] * win[i])` over the last `taps`
inputs (i32, wrapping, so the order of summation is irrelevant), `pred = s >= 0 ? (dot << s) >> 14 :
dot >> (14 - s)` (i64). After each sample, if `e != 0`: for every tap `w[i] = max(-32767,
saturating_i16(w[i] + sign(e) * ((bucket[i] << 8) >> k)))`, where `bucket[i]` is the bucketed
magnitude of window input i, computed when that input was pushed: with m = |in|, `b = 0` if in = 0;
else `4` if 48m > 4*avg, `2` if 48m > avg, else `1`; carried sign of in; then `avg += m - avg/16`
(truncating division). The window is pre-seeded from the stored `hist` re-quantised with the current
frame's s (their buckets are recomputed, advancing `avg`, at the start of each subframe).
A subframe without the flag (or without Fixed/LPC/Cross type) still advances the filter: the decoder
runs the same update on its recovered residual r (output discarded) with `s = clamp(bitlen(mean) - 9,
-16, 31)`, where `mean = floor(sum(min(|r[n]|, 2^40)) / N)` over the subframe's N recovered residuals and
`bitlen(0) = 0` (`stage2::Params::for_block(r, taps, k, 9)`); if any |r| > 2^40, or the subframe has no
residuals, the filter is left unchanged. `src/stage2.rs` (`Carried`) is the reference for these rules.

**OLS chunk (cfg 3).** Two channels, no frames. The payload after the config byte is blocks of
4096 frames (the last one shorter), each holding for channel 0 then channel 1: `[flag 1 bit][s_plus_16
6 bits if flag]` then a partitioned-Rice residual of the block, and the chunk ends byte-aligned with no
trailing bytes. Each channel has a cfg-1 carried filter (512 taps, k = 8); when bits_per_sample <= 16 it
uses the "wide" variant: bucket thresholds `4` if 96m > 16*avg, `2` if 96m > 6*avg, else `1`, and
`avg += m - avg/32` (for 17-24 bits the cfg-1 buckets and `avg/16` apply unchanged). The
carried output (or the raw residual if flag = 0, in which case the filter advances as above) is the
OLS residual, and the samples are `sample = ols_residual + ols_prediction`. The OLS predictor is
backward-adaptive in IEEE-754 binary64 and is specified by `src/ols.rs` (n = m = 16, lambda = 0.998,
solve every 16 samples, regularisation 1.0, Cholesky, binary64 +, -, *, /, sqrt only, no FMA; every sum
receives its terms in the order of the plain sequential form, and an implementation may compute independent
elements side by side, e.g. in SIMD lanes, but must not reorder the terms of any sum). The predictor state and its sample history start at zero in every chunk,
and the OLS statistics (covariance, cross-correlation and the solve counter) are not updated for the
first 16 frames of a chunk; with the first solve after 16 updates, at least the first 32
predictions of each channel are 0. **Unlike every other part of this format, cfg 3 is therefore bit-exact only
on platforms with correctly rounded binary64 arithmetic**; a decoder must not use fused multiply-add or
reassociate. A decoded sample outside the bit depth's range rejects the chunk. The value map (if any) is
applied to the decoded channels afterwards.

### FEC parity block

```
[sync: "FPAR", 4 bytes] [count: u32 LE] [m: u16 LE] [shard_len: u32 LE] [hdr_crc: u32 LE]
[table[0]: frames(u32 LE) + bytes(u32 LE) + crc(u32 LE)]   -- 12 bytes, one per covered chunk
...
[table[count-1]]
[shard_crc[0..m]: u32 LE each]
[shard[0..m]: shard_len bytes each]
```

`count` (1..=60000) is how many of the immediately preceding data chunks the block covers: all of
the file's chunks by default (a whole-file block), or `--fec=N` per block (the last block may be
shorter). `m` (1..=1024, at most `count`) is the number of parity shards; `shard_len` is the longest
covered payload rounded up to an even number. `table[i]` is a redundant copy of chunk `i`'s own
`frames`/`bytes`/`crc`; `hdr_crc` is the CRC-32 of `count`, `m`, `shard_len`, the table and the shard
CRCs (checked only when recovery or resync needs the block).

**Coding.** Reed-Solomon over GF(2^16) (polynomial 0x1100B, generator 2). Each chunk's payload is
read as little-endian 16-bit symbols, zero-padded to `shard_len`. Shard `j` is the symbol-wise sum
over chunks `i` of `c(j,i) * chunk_i`, where `c(j,i) = 1 / (j XOR (1024 + i))`. This Cauchy matrix
has every square submatrix invertible, so **any `e <= m` damaged chunks are rebuilt from any `e`
intact shards** -- the most any code of this redundancy can do. The size cost is about `m / count`
of the compressed size (more when chunk sizes vary, since every shard is as long as the longest
chunk).

**Recovering.** After a chunk fails its own CRC, the decoder checks every chunk of its block against
the table (a chunk whose payload matches the table but whose own header CRC field is damaged is
simply accepted), takes shards whose own CRC holds, forms `e` syndromes from them and the intact
chunks, inverts the `e x e` Cauchy submatrix, and checks every rebuilt chunk against its table CRC.
Anything that does not verify -- more damaged chunks than intact shards, damaged table -- is
reported as a failure, never decoded to wrong audio. The result is cached per block,
so a file with many damaged chunks reads and solves each block once.

**Resync.** If the ordinary chunk walk fails (a damaged chunk header: sync, length), the decoder
scans for `FPAR` syncs whose `hdr_crc` holds and chains the blocks: each must begin exactly where the
chunks its table lists end, the chain must cover the whole file, and the frame counts must sum to
`total_frames`. The chunks are then located from the tables alone. Damage to the parity block's own
header or table, or a chain that does not close, remains a hard rejection.

**Cost discipline.** Locating parity blocks during the walk reads `count`, `m`, `shard_len` and the
table (O(count)). Neither `hdr_crc` nor the shard CRCs are checked against the real bytes then: that
would cost O(file size), and a healthy file must not pay for it (decode speed priority).

For a known-length stream, the last block's parity comes after the point where `total_frames` is
accounted for; a reader allows exactly one more `FPAR` block there before requiring the stream to
end.

FEC also never applies to `encoder::StreamEncoder`/`decoder::decode_stream` output (the
never-buffer-more-than-one-chunk streaming path) -- feeding an FEC-enabled file to `decode_stream`
fails cleanly with a sync-mismatch error on the first `FPAR` block it meets, not a silent
misinterpretation.

### Metadata block (optional tags + embedded artwork)

Always present (an empty one if there's nothing to say), so the format never needs a separate
presence flag. `src/metadata.rs` is the authoritative implementation; field layout mirrors two
existing, well-understood conventions rather than a bespoke scheme -- Vorbis
Comment for text tags, FLAC's `METADATA_BLOCK_PICTURE` field-for-field for artwork -- which does
*not* by itself mean existing Vorbis/FLAC-aware tools read `.fak` files, since the surrounding
container (this table, these CRCs) is this project's own.

```
[body_len: u32 LE] [body: body_len bytes] [crc: u32 LE, CRC-32 of body]

body:
    vendor_len   : u32 LE
    vendor       : vendor_len bytes, UTF-8
    tag_count    : u32 LE
    tag[i]       : u32 LE length + that many UTF-8 bytes, once per tag -- the whole "KEY=VALUE"
                   string (Vorbis Comment convention: freeform keys, not a fixed enum; repeated
                   keys, e.g. multiple ARTIST=, are allowed)
    picture_count: u32 LE
    picture[i]   : kind (1 byte, FLAC picture-type values; unrecognized values are preserved
                         round-trip, not rejected)
                   mime_len (u32 LE) + mime (that many UTF-8 bytes)
                   desc_len (u32 LE) + description (that many UTF-8 bytes)
                   width, height, depth, colors (u32 LE each; colors == 0 for non-palette images)
                   data_len (u32 LE) + data (that many bytes, the raw image file)
    catalog_len  : u32 LE
    catalog      : catalog_len bytes, UTF-8 (whole-disc UPC/EAN; empty if no cue sheet)
    track_count  : u32 LE -- 0 means "no cue sheet" (`None`, distinct from a cue sheet present
                   with zero tracks only in the in-memory `Metadata` struct, not on disk -- both
                   serialize identically, since a cue sheet with no tracks carries no information)
    track[i]     : number (1 byte)
                   isrc_len (u32 LE) + isrc (that many UTF-8 bytes; empty if none)
                   index_count (u32 LE)
                   index[j]: number (1 byte, 0=pre-gap/1=track start/2+=sub-index) +
                             sample_offset (u64 LE, sample-frames from the start of the stream --
                             this format's native unit, not CD-style 1/75s frames)
    mask_present : 1 byte, 0 or 1 -- NOT a sentinel: a source WAV can legally declare
                   dwChannelMask == 0 ("no defined layout"), which must stay distinct from no mask
                   field being present at all
    channel_mask : u32 LE -- the source WAV's WAVE_FORMAT_EXTENSIBLE dwChannelMask (which physical
                   channel is front-left/front-right/LFE/etc.), 0 if mask_present == 0
    float_present: 1 byte, 0 or 1 -- iff 1, the source was 32-bit float PCM,
                   reduced to plain integer PCM by an exact fixed-point grid mapping before encoding
                   (`src/floatpcm.rs`); everything below is present only when float_present == 1
    scale_exp    : 1 byte -- a grid-mapped sample's exact real value is `k / 2^scale_exp`, where `k`
                   is what the chunk/frame/subframe layer actually encoded as that sample
    exception_count: u32 LE -- samples that were not exactly on the grid at scale_exp (or were
                   `NaN`/`Inf`/`-0.0`, none of which have a finite grid value), stored verbatim
    exception[i] : channel (u32 LE) + index (u64 LE, sample-frame index within that channel, over
                   the whole stream) + bits (u32 LE, the exact original float32 bit pattern) -- the
                   corresponding sample the chunk/frame/subframe layer encoded for this position is
                   an unused placeholder (0), ignored on decode once the exception is applied
```

Sanity bounds (checked before any declared length is trusted for allocation):
`tag_count <= 4096`, every string `<= 2^16` bytes (catalog/ISRC strings additionally `<= 64` bytes),
`picture_count <= 64`, every picture's `data_len <= 64 MiB`, `track_count <= 999`, every track's
`index_count <= 100`, `exception_count <= 2^22`. A decoder must additionally reject any index's `sample_offset` exceeding the
stream header's `total_frames` (checked once the metadata block and header are both parsed, and
skipped for a `TOTAL_FRAMES_UNKNOWN` stream -- there is no declared total yet to check against) and,
for the block as a whole, one that under- or over-runs its declared `body_len`, a `crc` mismatch, or
invalid UTF-8 in any string field. `decoder::decode_stream` (the streaming reader, below) additionally
caps `body_len` itself at 128 MiB before allocating a buffer for it -- the in-memory reader gets this
for free from the real byte slice's own length, but a streaming reader has no such bound until the
bytes actually arrive, so an unbounded declared length is a real allocation-bomb vector there.

CLI: `fak encode ... -T/--tag KEY=VALUE` (repeatable) and `--picture [TYPE:]PATH` (repeatable; `TYPE`
is `front`/`back`/`artist`, defaulting to front cover, and comes *before* the path deliberately --
`PATH:TYPE` would collide with a Windows path's own drive-letter colon), plus `--cuesheet PATH` to
embed a standard `.cue` sheet for a single-file disc image. `fak edit` changes tags, pictures and the
cue sheet of an existing file (audio bytes copied unchanged); `fak info` prints stream properties,
tags, pictures and the cue sheet without decoding audio; `fak picture <file> <n> <out>` writes a
picture's raw bytes back out. The CLI does not parse image dimensions/color depth
(`width`/`height`/`depth` are written as 0); a known gap, not a format limitation.

**Cue sheets: the `CUESHEET` tag and the binary cue sheet.** A cue sheet is stored twice,
by design. The `CUESHEET` tag (key matched case-insensitively) holds the sheet's text verbatim --
the convention foobar2000 and EAC-made FLAC files use, and the only place per-track `TITLE`/
`PERFORMER` survive. The binary cue sheet above holds the same track/index points as sample-frame
offsets, for tools that cut or seek by track without parsing text (`fak decode --track`). **When
the tag is present it is authoritative:** every writer in this project (`fak encode`/`fak edit`,
the foobar2000 component via `fak_rewrite_metadata`) re-derives the binary cue sheet from it
(`Metadata::sync_cue_sheet`), and removing the tag removes the binary copy. A file with only the
binary form (no tag) is valid; readers then generate `.cue` text from it (`fak cue`,
`Metadata::cue_sheet_text`; offsets that are not whole CD frames round down to one).
`fak::metadata::parse_cue_text` recognizes `CATALOG`/`FILE`/`TRACK NN <type>`/`INDEX NN MM:SS:FF`/
`ISRC`, converting CD timestamps (1/75 s) to sample-frames at the stream's sample rate; `TITLE`,
`PERFORMER`, `REM` and unknown lines are ignored there (they live on in the tag). It rejects a sheet
that cannot describe one stream: more than one `FILE`, a track without `INDEX 01`, track numbers not
strictly ascending, or index points going backwards; and writers reject an index point past the end
of the audio. A track spans its `INDEX 01` to the next track's `INDEX 01` (a pre-gap, `INDEX 00`,
plays as the end of the previous track), or to the end of the stream.

**CD tags.** Tags, not format fields: any reader may ignore
them. Opt-in: with `--cd-tags`, when the source is 16-bit 44.1 kHz stereo and its cue sheet's
positions and the stream's length are all whole CD sectors (588 frames, the image starting at
sector 0), `fak encode` adds the disc's identifiers and checksums (`fak::cdrip::disc_tags`),
unless the user set the same key:
`DISCID` (freedb, 8 hex digits), `MUSICBRAINZ_DISCID`, `ACCURATERIPID`
(`NNN-id1-id2-freedb`, CUETools' form), `FAK_CD_TOC` (track starts and lead-out in sectors,
`:`-separated, CTDB's lookup syntax), `FAK_ACCURATERIP_V1` / `FAK_ACCURATERIP_V2` (one 8-hex-digit
checksum per track, space-separated, track order) and `FAK_CTDB_CRC`. They are computed at read
offset 0 from the audio as stored; they say what the audio is, not that it matches any database
(`tools/cd/cd_lookup.py` checks that). Other sources get none of them. `fak edit --cd-tags`
writes or recomputes all seven from the embedded sheet (decoding the audio; stale values are
removed even when the sheet is not an exact CD layout). A file that has any of the four `FAK_`
tags has opted in: `fak edit --cuesheet` and the foobar2000 component (when a sheet's track points
move) then recompute them without being asked, and removing the sheet removes the `FAK_` ones (the
other three may come from a ripper, which is also why only the `FAK_` ones mark the opt-in). In
every writer, a key the same command sets explicitly is left to it.

Chunk length is an encoder choice, not a format constraint. The reference encoder defaults to 1
second (rounded up to a multiple of 4096 sample-frames; the last chunk may be shorter) -- lowered
from an earlier 10-second default to keep seek latency low; a block-mode
chunk costs only its 16-byte inline header. `--chunk-seconds` overrides it per encode.

### Stream header (56 bytes, fixed)

| offset | size | field | notes |
|---|---|---|---|
| 0 | 4 | magic | ASCII `"FAK1"` (the version byte follows) |
| 4 | 1 | version | `21` (a decoder rejects every other value) |
| 5 | 1 | channels | 1-255 |
| 6 | 1 | bits_per_sample | 8, 16, 24, or 32. For a grid-mapped float32 source (metadata block's `float_info`), this is the mapped *integer* domain's width, not literally 32 -- often narrower for quiet content |
| 7 | 1 | mode | reserved: must be `0` (block-independent); any other value is rejected |
| 8 | 4 | sample_rate | u32 LE, must be nonzero |
| 12 | 8 | total_frames | u64 LE, sample-frames (not bytes, not samples) in the stream, or `TOTAL_FRAMES_UNKNOWN` (`u64::MAX`) for a genuinely unbounded/streamed source whose length isn't known yet |
| 20 | 32 | pcm_hash | SHA-256 of the decoded PCM, canonical layout in `sha256::pcm_digest` (interleaved samples, `ceil(bits_per_sample/8)` little-endian bytes each). Detection-only; not checked by `decode`/`decode_full`, only by `decoder::verify` |
| 52 | 4 | header_crc | CRC-32 (reflected, poly 0xEDB88320) of bytes 4..52 |

A decoder must reject: bad magic, unsupported version, channels==0, bits_per_sample not in
{8,16,24,32}, `mode` != 0, sample_rate==0, or a header CRC mismatch. This applies before the decoder looks at `mode` to decide
how to interpret anything past the header.

32-bit sources also never carry a value-map section (below) or a subframe's
`ltp` bit set: the reference encoder never emits either for `bits_per_sample`==32, since their
existing margins (`valuemap.rs`'s used-value bitset, `ltp.rs::LIMIT`) were sized for <=24-bit
containers and were not re-derived rather than trusted unaudited. A decoder still accepts a stream
that has them (nothing in the wire format forbids it), subject to the same defensive bounds as
always.

### Frame

A frame is a byte-aligned unit: a bit-packed payload, padded with zero bits to the next byte
boundary. It has no sync word and no CRC of its own: frames are only ever parsed
inside a chunk whose CRC-32 has already been checked, and any error rejects the whole chunk, so a
per-frame check added nothing but bytes and a second pass over them.

```
payload (bit-packed, byte-aligned by padding):
    len_code        : 3 bits. 0..=6: frame_frames = 256 << len_code (256..=16384);
                      7: followed by frame_frames_minus_1 : 20 bits (frame_frames 1..=2^20).
                      Any length is valid and a decoder must not assume one: the reference
                      encoder picks 1024..=16384 per frame by default, or
                      DEFAULT_BLOCK_SIZE=4096 throughout with `--level fast` (a chunk's last frame
                      may be shorter either way, and then usually takes the escape)
    if channels == 2:
        stereo_mode : 2 bits (0=LeftRight, 1=MidSide, 2=LeftSide, 3=SideRight)
        subframe A  : see below
        subframe B  : see below
    else:
        subframe[c] : see below, once per channel, independently (no cross-channel decorrelation
                      for channel counts other than 2 in this version)
    [zero padding to next byte boundary]
```

A decoder reads frames until the running total equals *this chunk's own* declared frame count (from
its inline chunk header, not the stream-wide `total_frames`); a frame that would overrun that count
is rejected. `frame_frames` outside `1..=2^20` is rejected before it is used for any allocation.

Stereo channel bit depths follow FLAC's convention: a `side = l - r` channel needs one more bit
than the container's `bits_per_sample`; `mid = (l+r) >> 1` and plain `left`/`right` need exactly
`bits_per_sample`. The decoder derives each subframe's bit depth from `stereo_mode` alone — it is
not transmitted separately.

### Subframe

```
wasted      : 5 bits  (k = number of common trailing zero bits removed before coding; see below)
type        : 3 bits  (0=Constant, 1=Verbatim, 2=Fixed, 3=Lpc, 4=Palette, 5=PaletteRle,
                       6=Cross; 7 invalid)
if type == Cross:
    inner   : 3 bits  (2=Fixed or 3=Lpc; anything else invalid), then that type's fields below up
              to and including its warmup, then the cross-channel fields, then its residual:
    ref     : 8 bits  (index of an earlier subframe in this frame; must be < this subframe's index)
    source  : 1 bit   (0 = the reference's samples, 1 = its predictor residual)
    lag0    : 6 bits  (first lag, stored as lag0 + 32, so -32..=31)
    taps_minus_1: 4 bits (taps = value + 1, 1..=16)
    xprecision_minus_1: 4 bits (tap width q = value + 1; a decoder rejects q outside 3..=16)
    xshift  : 5 bits
    xcoeffs : `taps` raw signed coefficients, each q bits
if type == Fixed:
    order   : 3 bits  (0..=4; 5..=7 invalid)
    if from_history (see "Predictor history"):
        residual: partitioned-Rice-coded stream of frame_frames values (see below)
    else:
        warmup  : `order` raw signed samples, each (bits_eff - wasted) bits, two's complement
        residual: partitioned-Rice-coded stream of (frame_frames - order) values
if type == Lpc:
    order_minus_1: 5 bits  (order = value + 1, so order is 1..=32)
    shift   : 5 bits  (0..=31, the fixed-point binary-point position, see "LPC" below)
    precision_minus_1: 4 bits (coefficient width p = value + 1; a decoder rejects p outside 3..=16)
    coeffs  : two groups, coefficients 0..min(2, order) then the rest (if any); each
              group is a 4-bit Rice parameter kc followed by its coefficients as Rice codes (kc) of
              their zigzag values (unary quotient bounded at 2^17). Every coefficient must lie in
              -2^(p-1)..2^(p-1)-1; a decoder rejects one that does not
    if from_history:
        residual: partitioned-Rice-coded stream of frame_frames values
    else:
        warmup  : `order` raw signed samples, each (bits_eff - wasted) bits
        residual: partitioned-Rice-coded stream of (frame_frames - order) values
Every "residual" above (Fixed, Lpc, and so Cross) is preceded by the stage-2
header (see "Stage-2 adaptive prediction" below):
    stage2  : 1 bit  (0 = off: the residual follows as is)
    if stage2 == 1:
        s2_taps_code : 2 bits (taps = 16, 32, 128, 256)
        s2_k         : 4 bits (1..=15; 0 invalid)
        s2_s_plus_16 : 6 bits (s = value - 16; a decoder rejects s > 31)
and then by the long-term prediction header (see "Long-term prediction" below):
    ltp     : 1 bit  (0 = off)
    if ltp == 1:
        ltp_lag      : 11 bits (T = 32..=2047; a decoder rejects T < 32)
        ltp_k_code   : 2 bits (K = 1, 3, 5, 9 taps)
        ltp_taps     : K raw signed 7-bit values g[0..K) (two's complement, -64..=63)
if type == Palette:
    count_minus_2 : 4 bits  (table size = value + 2, so 2..=17; decoder rejects 17 since
                    MAX_PALETTE=16 -- see "Palette" below)
    table   : `count` raw signed values, each (bits_eff - wasted) bits
    indices : frame_frames values, each ceil(log2(count)) bits, indexing into `table`
              (decoder rejects any index >= count)
if type == Verbatim:
    samples : frame_frames raw signed samples, each (bits_eff - wasted) bits
if type == Constant:
    value   : 1 raw signed sample, (bits_eff - wasted) bits
```

`wasted` must be strictly less than `bits_eff` (the channel's bit depth as derived above); a
decoder rejects `wasted >= bits_eff`. All stored/predicted values are computed on the
*right-shifted* (by `wasted` bits) sample sequence; after reconstruction the decoder left-shifts
by `wasted` to restore the original magnitude. This is exact because every sample in the block is,
by construction, a multiple of `2^wasted` (see "Wasted bits" below).

Fixed predictors (orders 0-4) are FLAC's classical integer differencing family:

```
order 0: pred = 0
order 1: pred = x[-1]
order 2: pred = 2*x[-1] - x[-2]
order 3: pred = 3*x[-1] - 3*x[-2] + x[-3]
order 4: pred = 4*x[-1] - 6*x[-2] + 4*x[-3] - x[-4]
```

computed in i64 (order 4's coefficients sum to 15 in absolute value, far short of
overflowing i64 even at 25-bit samples).

### Predictor history

Frames within a chunk are decoded strictly in order (a chunk is already the unit of independence:
seeking, threading and FEC all work on whole chunks, and any error in a chunk rejects all of it), so
a frame's predictor can start from samples already decoded earlier in the same chunk instead of
paying for `order` verbatim warmup samples.

For each subframe, `history` is the last `min(32, samples decoded so far in this chunk)` samples of
*that subframe's own channel representation*, computed from the already-reconstructed output
channels, and right-shifted by *this* subframe's `wasted` (arithmetic shift, i.e. floor; the history
need not be a multiple of `2^wasted` -- it is only ever an input to prediction, identically on both
sides):

- mono / 3+ channels: the channel's own previous samples;
- stereo: `L`, `R`, `M = (L+R)>>1` or `S = L-R` of the previous samples, according to *this*
  frame's `stereo_mode` (so a frame may use a different representation from the one before it).

`from_history` is true iff `order > 0` and `history.len() >= order`. It is not transmitted: both
sides derive it. When true, the warmup is `history[len-order..]` (oldest first) and every one of the
frame's `frame_frames` samples is predicted and Rice-coded. A chunk's first frame always has an empty
history, so it codes its warmup verbatim verbatim; nothing crosses a chunk boundary.

### LPC

Higher-order prediction: `pred = (sum(coeffs[j] * x[-1-j] for j in 0..order)) >> shift`, computed
in an i128 accumulator (order up to 32, coefficients up to 16-bit signed, history samples up to
`2^51` in magnitude in the worst hostile case (see "Defensive bound") — at most `2^5 * 2^15 * 2^51 =
2^71`, which can't overflow i128 but *could* overflow i64, hence i128). `shift` and `coeffs`
are found at encode time by windowed-autocorrelation + Levinson-Durbin (float) then quantized to
fixed point with error feedback (each coefficient's rounding error carried into the next, reducing
quantization noise versus rounding each independently). The order search quantizes at 14 bits; the
winner is then requantized at neighbouring precisions (`encoder::refine_precision`) and the
cheapest is written with its own `precision` field. `shift` is chosen per precision so the largest
coefficient uses the available bits. From a fixed candidate-order list
(`{1,2,4,6,8,12,16,24,32}`, `src/encoder.rs::LPC_ORDER_CANDIDATES`), the encoder first ranks orders
analytically using the Levinson-Durbin recursion's own predicted-error sequence (a free byproduct —
no residuals materialized), then quantizes and computes real integer residuals + cost for only the
top `LPC_SHORTLIST` (3) of them (this shortlisting is an encode-
speed/compression tradeoff, measured at <0.5% cost on every real file tested). Whichever of those —
alongside the fixed predictors and Verbatim/Constant/Palette — has the lowest real
(quantized-integer-residual, not estimated) cost is written. Unlike the fixed predictors, LPC
coefficients are per-block side information, so the cost comparison charges their exact coded size
(`lpc::coeff_bits`) plus 9 bits for them (plus `order * bits_eff` of verbatim warmup when not `from_history`) before comparing
against the alternatives.

### Cross-channel prediction

A `Cross` subframe codes a Fixed or LPC subframe exactly as above, except that its residual is net of
a cross-channel term over a *reference*: subframe `ref` of the same frame, which is decoded first.
With `n = frame_frames` and `src` the reference's signal over the frame (length `n`):

```
term(t) = floor( sum_{k < taps} xcoeffs[k] * src[clamp(t + lag0 + k, 0, n - 1)] / 2^xshift )
x[t]    = own_prediction(t) + term(t) + residual[t]         (in this subframe's wasted domain)
```

- `source = 0`: `src` is the reference's decoded samples with its wasted bits restored (for stereo,
  the first subframe's channel representation -- L, M or S by `stereo_mode` -- before the stereo
  transform is undone).
- `source = 1`: `src` is the reference's *predictor residual*: for a Fixed/LPC subframe the Rice-
  coded values, placed at the frame positions they cover (a subframe whose warmup was stored
  verbatim has no residual at its first `order` positions: those are 0); for a `Cross` reference, its
  own predictor's residual, i.e. with its cross-channel term added back; for any other type, all 0.
- `t` ranges over the positions the residual covers (all `n` when `from_history`, else
  `order..n`); lags into the reference's "future" are valid, the whole reference being decoded.
  Indices are clamped to the frame, so no sample outside it is read.

The decoder adds `term` to the decoded residual and reconstructs with the ordinary Fixed/LPC
recursion (history, warmup and the `2^48` bound unchanged). Integer ranges: a decoder rejects a
reference signal with any `|src| > 2^40` (legitimate ones are below `2^27`), so with `|xcoeffs| <=
2^15` and at most 16 taps the sum is below `2^59`; a Rice-decoded residual is below `2^62` in
magnitude (`k <= 30`, quotient `< 2^32`, escapes `<= 56` bits, widened from 40 for 32-bit
containers), so `residual + term` fits `i64`.

The reference encoder tries, for the second subframe of each stereo mode and for each channel after
the first of a multichannel frame (referencing the earlier channel whose residual correlates best),
two candidates per lag window: `source = 1` with taps fitted to the subframe's own residual, and
`source = 0` with the LPC coefficients re-solved jointly with the taps (exact least squares). Windows
are `(lag0, taps) = (-2, 5)`, and `(-5, 11)` only when the narrow one already saves 3%; a candidate
must save 1% of the subframe's estimated bits to be used (each costs the decoder an FIR pass).

### Stage-2 adaptive prediction

When a subframe's `stage2` bit is set, the Rice-coded values `e[t]` are not the residual but what is
left after a second, sample-adaptive prediction of the residual from its own past. The decoder
recovers the residual `r[t]` (the value the sections above call "residual", i.e. before the
cross-channel term is added back) in order, t = 0, 1, ...:

```
state per subframe: weights w[0..taps) and window x[0..taps) of i16, all 0 at t = 0
                    (x[taps-1] is the newest input)
dot    = sum_i w[i] * x[i]        each product exact in i32, the sum wrapping modulo 2^32
pred   = s >= 0 ? (dot << s) >> 14 : dot >> (14 - s)       (i64, arithmetic shifts)
r[t]   = e[t] + pred              a decoder rejects |r[t]| > 2^40
g      = sign(e[t])               (-1, 0 or +1)
w[i]   = sat16(w[i] + g * (x[i] >> k))  for every i   (arithmetic shift; sat16 clamps to i16)
x      = window shifted by one: drop x[0], append sat16(s >= 0 ? c >> s : c << -s),
         c = clamp(r[t], -2^40, 2^40)
```

The encoder runs the same recursion with `e[t] = r[t] - pred`. Everything is integer and exactly
specified (the wrapping sum is what `pmaddwd`/`smlal` produce, and its value does not depend on
summation order), so every implementation reconstructs identically; the reference decoder has a
scalar version and AVX2/NEON versions tested against it. The state never crosses a subframe, so
frames stay independently decodable. The reference encoder chooses `s` so the residual's mean
magnitude is about 2^9, tries a few `(taps, k)` and keeps the stage only when the subframe gets
smaller with the 12 extra header bits. Because the stage roughly
doubles decode time, the reference encoder uses it only at level `insane`; other levels write
the bit as 0 (one bit per subframe).

### Long-term prediction

`LIMIT` (below) was sized so this stage's own dot product stays exact in 32-bit arithmetic at
<=24-bit containers; not re-derived for 32-bit samples' wider residuals, so the reference encoder's
search never runs at `bits_per_sample`==32 and every such subframe's `ltp` bit is 0.

When a subframe's `ltp` bit is set, the Rice-coded values `c[n]` (n = 0 .. m-1, the subframe's
residual length) are what is left after predicting each value from K values around lag T earlier
in the same subframe. The decoder undoes this first, before stage 2, recovering `e[n]` (the values
stage 2 reads, or the residual itself when stage 2 is off), in order n = 0, 1, ...:

```
h    = (K - 1) / 2
e[n] = c[n]                                                       for n < T + h
e[n] = c[n] + ((sum_{j<K} g[j] * e[n - T - h + j] + 16) >> 5)     for n >= T + h
                                          (exact integers; >> is an arithmetic shift, i.e. floor)
a decoder rejects the subframe if any |e[n]| > 2^20
```

The range rule makes every intermediate exact in 32 bits (|sum| <= 9 * 64 * 2^20 + 16 < 2^30).
Since T - h >= 28, the predictions of a run of `T - h` consecutive values read only values before
that run, so a decoder may compute a whole run at once (the reference decoder does, in i32 lanes).
Nothing crosses a subframe boundary. The reference encoder finds up to 6 lag candidates from the
residual's autocorrelation (FFT with `detmath` twiddles, so the choice is platform-independent),
fits the taps from the autocorrelation, and keeps the best (T, K) only when the subframe gets
smaller including its 14 + 7K header bits, and, at `normal` and `max`, only when it saves at
least 0.05 bits per value (level `fast` does not search it and writes the flag as 0), since every subframe that uses
the stage costs the decoder a pass over its residual. Both are
encoder choices; a decoder accepts any valid header.

### Defensive bound on predictor feedback (both Fixed and LPC)

Because residuals feed back into the predictor's history, a corrupted or adversarial residual
(e.g. from a maximal-width Rice escape partition) could in principle compound across samples
toward an integer overflow before the frame's CRC is even checked (CRC verification happens only
after the whole frame is parsed). Both `predictors::reconstruct` and `lpc::reconstruct` compute
each reconstructed sample in an i128 accumulator and reject the frame immediately (a decode
error, not a panic or silent wraparound) if any reconstructed sample's magnitude exceeds
`2^48` — far beyond anything a legitimate 25-bit- or, for 32-bit sources, 33-bit- (32-bit samples, widened
side channel) sample stream could ever produce, but small enough to catch runaway growth within
the first few affected samples rather than letting it run toward i128's true limit. The Rice
escape path's raw-value width field is separately capped to 56 bits at decode time for the same
reason (`src/rice.rs::MAX_ESCAPE_WIDTH`, widened from 40).

The predictor history reuses already-bounded output: every Fixed/LPC subframe's samples are
within `2^48`, every other subframe type's within the container's own widened range (`2^25` at
<=24-bit, `2^33` at 32-bit), so a stereo recombination (`L`, `R`, then `M`/`S` of those
for the next frame's history) stays well inside `2^48` either way. A hostile stream therefore can't
grow history from frame to frame: each frame's own reconstruction is re-checked against `2^48`
regardless of how large its history is.

### Value map

The reference encoder's detector assumes a container of at most 24 bits (its used-value
bitset is sized off the chunk's value range, infeasible at 32-bit) and never runs above that; a
32-bit chunk always takes `tag == 0`, below.

Every chunk payload starts with this section, byte-aligned, before the first frame:

```
tag : 1 byte. 0 = no channel is mapped (the whole section); 1 = a per-channel list follows
if tag == 1, for each channel in order:
    flag : 1 byte, 0 (not mapped), 1 (mapped) or 2 (mapped, with corrections)
    if flag >= 1:
        P : u64 LE
        C : u64 LE
    if flag == 2:
        count : u32 LE, 1 <= count <= the chunk's sample-frames
        len   : u32 LE, bytes of the correction list that follows
        list  : len bytes, bit-packed MSB-first like frames:
                kg : 5 bits, Rice parameter of the position gaps (0..=24)
                kv : 5 bits, Rice parameter of the values (0..=24)
                count times: gap (Rice kg) then zigzag(e) - 1 (Rice kv)
                zero-padded to a byte; the list must end exactly at len
```

Correction `i`'s position is `gap_0` for the first and `position_{i-1} + 1 + gap_i` after that
(strictly increasing), and must be below the chunk's sample-frame count; its value `e` is
`unzigzag(v + 1)`, never 0, `|e| <= 2^26`. A decoder must reject any of these violated, and a
Rice code whose quotient exceeds its bound (`(frames >> kg) + 1`, `(2^27 >> kv) + 1`).

The chunk's frames then decode exactly as described below, but for a mapped channel what they
produce is `k`, not the output sample. After all frames of the chunk are decoded, each mapped
channel's samples are replaced by

```
x = clamp(floor((k * P + C) / 2^32), lo, hi)     lo = -2^(bits_per_sample-1), hi = 2^(bits_per_sample-1) - 1
```

except at a listed correction position, where `x = clamp(floor((k * P + C) / 2^32) + e, lo, hi)`.

with `k` first clamped to `[-2^25, 2^25]` (any `|k|` beyond that maps outside every container
range, since `P > 2^32`; the clamp only keeps the arithmetic in range on hostile input). Exact i64
form, used by `valuemap::ValueMap::apply` and tested against the i128 formula: with `P = q*2^32 + f`
and `C = cq*2^32 + cf`, `x = q*k + cq + ((k*f + cf) >> 32)` (arithmetic shift), then the clamp.
Stereo decorrelation, wasted bits and predictor history (all per frame) operate on `k`: the map is
a per-chunk channel transform applied after them, like a gain stage at the very end.

A decoder must reject: a tag other than 0 or 1; a flag other than 0, 1 or 2; a truncated section;
`tag == 1` with no channel mapped; `P <= 2^32` (a gain of at most 1 cannot be a lossless
coarsening); `C >= P` (any `C` can be reduced mod `P` by shifting `k`, so larger values are not
canonical). Nothing else about `P`, `C` or `k` can fail: the clamp makes every decoded value a valid
sample.

The map covers PCM whose values lie on a structure finer than wasted bits: a non-power-of-two
integer step (`x = 3k`), an offset lattice (`x = 256k + 128`), and a fixed gain above 1 applied to
lower-resolution material without dither (`x = round(k*G)`, e.g. 16-bit audio exported to a 24-bit
container at -1 dB). The clamp makes samples that were clipped at the container's extremes
representable. The reference encoder's detection (`valuemap::detect`) is not part of the format.

### Wasted bits

If every sample in a subframe's block is a multiple of `2^k` (common when audio that originated
at a lower bit depth was left-shifted into a wider container — FLAC calls this the same thing),
those `k` low bits carry no information and are removed before prediction/entropy coding, then
restored by a left-shift at decode time. `k` is capped so at least 1 bit of headroom remains (an
all-zero block is caught by the Constant case instead) and to 30 regardless. Measured effect: on
a synthetic 4-wasted-bit test signal this took the subframe from 7.07 bits/sample to 3.12
bits/sample — without it, those dead bits are paid
for in every residual.

### Palette

If a block's (post-wasted-bits-shift) samples use at most 16 distinct values, they can be coded as
fixed-width indices into a small per-block table instead of via prediction + Rice. This exists for
signals with *no* predictive structure but a small raw alphabet -- e.g. i.i.d. samples drawn from a
handful of fixed values, where consecutive samples carry no information about each other, so Fixed
and LPC prediction (and Rice-coding their residuals, a magnitude-based code assuming values cluster
near zero) cost the full bit depth per sample regardless of predictor order. An index code captures
the raw-value repetition directly. Found via a WavPack comparison on a synthetic adversarial test
file: took that file from 16.0 bits/sample (no compression) to
1.02. Real music essentially never has ≤16 distinct raw sample values in a 4096-sample block, so
this subframe type is inert on ordinary content (measured: zero effect within noise on every real
corpus file). The decoder rejects a declared table size > 16 (the 4-bit field can represent up to
17) and any index ≥ the actual table size, both only reachable via a corrupted/hostile stream.

### PaletteRle

The same palette table as above, but with the *index stream* run-length-coded instead of stored at
a flat `index_bits(count)` per sample: this closes a real remaining gap Palette has on its own,
found on the `square100` synthetic file (a low-frequency square wave: only 2 distinct values, but
long runs of ~220 samples between transitions) -- flat Palette still pays 1
bit/sample regardless of run length, wasting almost all of it on runs that don't change.

```
count_minus2   : 4 bits   (palette size 2..=16, same convention/bound as Palette)
palette values : count * eff_bits, signed
run_len_bits   : 5 bits   (bit width used for every run-length field below; 1..=31; 0 is rejected)
num_runs_minus1: 20 bits  (num_runs 1..=2^20, matches MAX_FRAME_FRAMES's own bound)
per run (num_runs times):
    index            : index_bits(count) bits
    run_length_minus1: run_len_bits bits   (actual run length = value + 1)
```

The encoder always computes *both* Palette's and PaletteRle's real cost from the same built table
and index stream, and emits whichever is smaller (`encoder.rs`) -- never a heuristic guess, so this
subframe type can only ever help or be a no-op relative to plain Palette, never a regression
(verified: `encoder::tests::palette_rle_does_not_regress_no_run_structure`). Measured effect: the
i.i.d. two-value file stays at 1.02 b/s (every run has length 1 there, so flat `Palette` wins the
internal comparison and gets emitted, exactly as before this subframe type existed), but
`square100` went from 0.516 b/s to **0.041 b/s** -- well
below WavPack's 0.328.

The decoder rejects: a declared table size > 16 (same as Palette), `run_len_bits == 0`, a declared
`num_runs` exceeding the subframe's actual sample count `n` (checked *before* being used to size
any allocation), any index ≥ the table size, any single run whose length would
overrun `n`, and a final run-length total that doesn't sum to exactly `n` (both overshoot mid-loop
and undershoot at the end are rejected).

### Partitioned Rice coding

Residuals are zigzag-mapped to non-negative integers (`zz(e) = (e << 1) ^ (e >> 63)`), then coded
in partitions:

```
partition_size_idx : 3 bits  (index into {32, 64, 128, 256, 512, 1024}; 6 and 7 invalid)
per partition (of `partition_size` values, the last may be shorter):
    param : a parameter code, see below: an index i in 0..=61, or 63 = escape
    if param == escape:
        width : 6 bits (a decoder rejects width > 40)
        per value: width raw bits of zz(e)
    else, with k = i >> 1:
        i even (a Rice code):  unary(zz >> k), then k raw bits of zz & (2^k - 1)
        i odd (a recursive Golomb-Rice code):
            if zz < 2^(k+1):  `1`, then k + 1 raw bits of zz
            else:             `0`, then the Rice code (above) of zz - 2^(k+1) with parameter k
    (every unary quotient is bounded at 2^32; a longer one is a corrupted stream)

Index 2k is Rice parameter k, index 2k + 1 the recursive code with parameter k, for k in 0..=30;
62 is invalid. The recursive code's first 3 * 2^k values all take k + 2 bits and each further 2^k
one more, so its scale sits between Rice k and Rice k + 1. A decoder reads it as one unary
quotient q (the leading `0` included), then k raw bits, or k + 1 when q = 0; the value is
low + ((q + (q != 0)) << k).

parameter code:
    first partition of the stream, or every partition before the first non-escape one:
        6 raw bits (the index, or 63)
    otherwise, relative to p = the last non-escape partition's index:
        `0`            -> p
        `10` sign      -> p + 1 (sign 0) or p - 1 (sign 1)
        `110` sign     -> p + 2 or p - 2
        `111` + 6 bits -> that index, or 63 = escape
    a result outside 0..=61 (other than the explicit escape) is invalid
```

The extra low bit of a recursive code with q = 0 is signalled by the code's first bit being `1`,
so a decoder knows the code's length from the first bit in parallel with the quotient's
leading-zero count.

The encoder chooses each partition's code greedily in stream order -- whichever of the eight
indices around the mean's Rice parameter, or the escape, is smallest including its own parameter
code after the previous choice -- and picks `partition_size` itself (once for the whole residual
stream) by real cost across all six candidates. A fixed partition size is a poor fit for a residual that is mostly zero with rare huge
spikes -- e.g. a near-perfectly-predicted square wave, where the predictor is exact except at each
edge -- because one spike in an otherwise-quiet partition forces every zero sharing it to pay for
a Rice parameter sized for the outlier. The escape path exists so a single pathological outlier that
*can't* be isolated by a smaller partition can't blow up the unary code arbitrarily; the decoder
additionally bounds unary-code length defensively regardless of what the
encoder would ever produce, since it must not trust the stream.

### Seeking

`decoder::seek` locates the chunk containing a target frame by walking the
self-delimited chunk sequence's inline headers (`format::locate_chunks`), decodes only that one
chunk (effectively O(1) here, since every frame inside already decodes independently), and returns it
plus the chunk's own starting frame so the caller can trim to the exact requested sample. This walk
costs the same as an up-front table would (every header is fixed-size and every payload is skipped
over by byte offset, never decoded) -- it works on any `.fak` file, at any chunk length.

## Streaming

`encoder::StreamEncoder` and `decoder::decode_stream` work over any `std::io::Write`/`std::io::Read`
(a real pipe or socket, not just an in-memory buffer), for a genuinely unbounded/live source where
the total length isn't known when encoding starts:

- `StreamEncoder::new` writes the header immediately with `total_frames = TOTAL_FRAMES_UNKNOWN`
  (`u64::MAX`) and the metadata block, then `push_chunk` encodes and flushes one chunk's inline
  header + payload at a time -- never buffering more than one chunk, never seeking backward on the
  sink. No whole-stream `pcm_hash` is written (there's no way to see every sample before encoding
  starts), so `decoder::verify`/`fak verify` refuse a `TOTAL_FRAMES_UNKNOWN` stream outright rather
  than comparing against a meaningless all-zero hash.
- `decoder::decode_stream` reads the header and metadata block, then reads and decodes one chunk at a
  time, calling back into the caller before the next chunk is even read. For a `TOTAL_FRAMES_UNKNOWN`
  stream, the only valid stopping point is the source itself running out (clean EOF exactly at a
  chunk boundary); anything else (EOF mid-header, mid-payload, or a corrupted field) is an error, not
  a silent short read.
- The same self-delimited chunk framing also works for an ordinary known-length file
  (`total_frames` set normally, as `encoder::encode_chunked` writes): `decode_stream` on such a file
  stops once `total_frames` is reached and then requires a clean end of input, rejecting trailing
  garbage exactly like the in-memory `locate_chunks` path does.

CLI: `fak stream-encode <in.wav> [options] | fak stream-decode <out.wav>` -- verified end-to-end
through a real OS pipe (not just in-process calls) on real corpus audio, both stream modes,
bit-exact against the source via the independent Python `wavio.py` oracle; a genuinely truncated
pipe (`head -c` cutting the stream mid-flight) is correctly rejected with a nonzero exit code rather
than producing a silently-incomplete WAV. The CLI subcommands exist only as a convenient way to
exercise the API from a shell; the point is that the *format and library* are
network-streamable -- any application embedding this codec (not just `fak.exe`) can push chunks to
a live `TcpStream`/socket as audio is produced, and a peer can start decoding and playing before the
stream ends, without either side ever seeing the whole file.

That network claim is verified, not just architecturally inferred from the `Read`/`Write` generic
bounds: `tests/network_stream.rs` runs `StreamEncoder`/`decode_stream` over a real
`TcpListener`/`TcpStream` loopback connection (a different code path from a pipe on every OS --
different buffering, different partial-read/partial-write behavior), confirming (1) bit-exact PCM
over the socket for both an unknown-length live source and an ordinary known-length file, and (2)
genuinely progressive delivery -- the receiver's decode callback fires as each chunk lands on the
wire (paced ~40ms apart by the sender), not all at once after the connection closes, which is the
actual thing "streamable over the network" needs to mean.

## Reference-encoder limitations (not limits of the format)

- A value map covers a chunk-channel with at most 10% of samples off its lattice (the format's
  corrections carry those); more scattered material, or a map whose source values are not dense
  near their densest point, is not detected by the reference encoder. The format itself allows any number of corrections up to the chunk length.

- The LPC order search is a fixed candidate list, not exhaustive, and the per-subframe coefficient
  precision is found by a greedy walk, not an exhaustive search (measured within 0.03%
  of exhaustive).
- The encoder's variable frame length is a binary tree of power-of-two lengths aligned within
  each chunk (1024..=16384), chosen by an analytic estimate by default; arbitrary cut points are
  allowed by the format but not searched.
- Cross-channel prediction references one earlier subframe of the same frame per
  subframe, with at most 16 taps; the reference encoder picks the reference among earlier channels by
  residual correlation (one candidate per channel), not by trying all of them, and the stereo
  transform is still one of the four fixed modes.
- A `TOTAL_FRAMES_UNKNOWN` stream has no whole-stream integrity hash and no seek index (by
  definition -- its length isn't known until it ends), so `fak verify`/`fak seek` only work on a
  normal known-length file. Re-encoding a streamed file with plain `encode` recovers both.
- The encoder does not parse embedded-picture files for width/height/color-depth; those fields are
  written as 0 unless set some other way.
- The `.cue` importer supports one `FILE` describing one continuous stream (this format's own
  model); a `.cue` sheet spanning multiple `FILE`s isn't representable and its extra `FILE` lines
  are silently ignored rather than rejected.
- FEC recovers up to `m` damaged chunks per parity block (any `m` intact shards suffice); more damaged
  chunks than intact shards, or damage to a parity block's own header or table, is a hard failure. It
  repairs chunk payloads and, through resync, damaged chunk headers, provided the parity blocks'
  headers survive. FEC never applies to `encoder::StreamEncoder`/`decoder::decode_stream` output --
  only the file-based `encoder::encode_chunked`/`decoder::decode_full`/`seek` path.
