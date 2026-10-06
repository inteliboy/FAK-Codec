# FAK Codec

FAK is a lossless audio codec: `decode(encode(x))` is bit-for-bit identical to the source PCM, and every
file stores a SHA-256 of the audio it was made from so that `fak verify` can prove it. This repository
contains the codec library, the `fak` command-line tool, the test suite and the bitstream specification.

- **Format:** version 21, specification 1.1.0 ([`docs/bitstream-spec.md`](docs/bitstream-spec.md)).
- **Language:** Rust (stable, edition 2021). The only runtime dependency is the optional `mimalloc` allocator.
- **License:** MIT OR Apache-2.0, at your option.
- **foobar2000:** the component is in a separate repository, [foo_input_fak](https://github.com/inteliboy/foo_input_fak).

## Compatibility

A decoder accepts exactly one format version (shown by `fak version`). Files written by an earlier
version (for example version 20) are rejected with an error, not misdecoded. Version 21 is the
current format; see the specification for what a version change means and what is guaranteed.

## What it handles

| | |
|---|---|
| Sample formats | 8-, 16-, 24- and 32-bit integer PCM; 32-bit IEEE float (exactly, see the specification) |
| Channels | 1 to 255 (stereo has dedicated decorrelation; other layouts are coded per channel with optional cross-channel prediction) |
| Container | WAV input and output (`WAVE_FORMAT_PCM`, `WAVE_FORMAT_IEEE_FLOAT`, `WAVE_FORMAT_EXTENSIBLE` with channel mask) |
| Metadata | Vorbis-comment style tags, embedded pictures (PNG/JPEG), embedded cue sheet (+ CD disc IDs and AccurateRip/CTDB checksums) |
| Seeking and streaming | Independent chunks (about one second by default): seek granularity is one chunk, and chunks decode in parallel |
| Error handling | Per-chunk CRC; optional whole-file Reed-Solomon parity (`--fec`, `--level archival`) that rebuilds damaged chunks |

## Platforms

Built and tested by the CI in this repository on every push:

| OS | Architecture |
|---|---|
| Linux | x86-64, aarch64 |
| macOS | arm64, x86-64 |
| Windows | x86-64 |

The decoder is deterministic and integer-exact, so a file decodes to the same PCM on every platform. The
single exception to "integer only" is the OLS chunk type of the format (chunk config 3 and 4, produced at
`--level insane`): it uses binary64 arithmetic and the specification requires correctly rounded IEEE
operations without fused multiply-add. In the development tree's CI, files produced on every platform are
decoded on a reference platform and compared with the source PCM. SIMD kernels (AVX2 on x86-64, NEON on aarch64) are selected at
runtime and are tested against a portable scalar implementation.

The encoder is not required to be reproducible: two encodes of the same input may differ in bytes (hardware
acceleration, thread scheduling), and every such file decodes to the same audio.

## Build

```
cargo build --release          # target/release/fak (fak.exe on Windows)
cargo test --release
```

`cargo build --release --no-default-features` uses the system allocator instead of mimalloc.

## Use

```
fak encode album.wav album.fak                       # level normal
fak encode --level max album.wav album.fak
fak encode --level archival album.wav album.fak      # insane + Reed-Solomon parity
fak decode album.fak album.wav
fak verify album.fak                                 # decode and check the stored SHA-256
fak info album.fak                                   # properties, tags, pictures, cue sheet
fak edit song.fak --set-tag TITLE=Better --picture front:cover.jpg
```

`fak help` lists every command and option.

## Compression levels

| Level | Description |
|---|---|
| `fast` | fixed 4096-sample frames |
| `normal` | frame lengths chosen by an estimate (default) |
| `max` | frame lengths chosen by trial encoding, about 4x the encode time of `normal` |
| `insane` | `max` plus adaptive prediction stages (a carried long filter and, for stereo up to 24 bits, an OLS predictor) |
| `archival` | `insane` plus Reed-Solomon parity |

Measured on one machine (x86-64, Windows 11), one run, every codec single-threaded, on 20-second excerpts
(starting at 30 s) of 12 real music recordings (5 at 16-bit, 7 at 24-bit; classical, spoken-word, electronic
and vocal recordings; 240 s of audio in total), against FLAC 1.5.0 at `-8`; all 48 decodes were bit-exact:

| Codec | Size vs FLAC `-8` (total bytes) | Encode speed | Decode speed |
|---|---|---|---|
| FLAC 1.5.0 `-8` | baseline | 233x realtime | 617x realtime |
| FAK `max` | -2.03% | 42x | 497x |
| FAK `insane` | -2.82% | 8x | 47x (37x on 24-bit) |

`insane` was smaller than FLAC on all 12 files (from -1.6% to -6.3%); the gain is larger on the 16-bit
files (-5.1% in total) than on the 24-bit ones (-2.4%). Treat this as an indication, not a benchmark: it is
12 excerpts, a single timing run and a single machine, and the files are mostly classical music. Results on
other material, and on full-length files, will differ.

## Correctness and robustness

- Every file carries the SHA-256 of its source audio; `fak verify` decodes the whole file and compares.
- The test suite (`cargo test --release`) covers round trips at every compression level, bit depths 8 to 32,
  float input, extreme sample values, chunk and frame boundary lengths, damaged and truncated files, files
  with damaged chunks that parity repairs, the stereo OLS chunk, and decoding of stored files from a
  previous build (`tests/data`) so that a decoder change cannot silently alter what an existing file means.
- The decoder treats its input as untrusted: sizes are bounded before allocation, arithmetic is checked, and
  malformed files return an error. The test suite includes randomized corruption of valid files.

## Specification

[`docs/bitstream-spec.md`](docs/bitstream-spec.md) specifies the container, the chunk and frame structure,
every predictor and entropy-coding step the decoder performs, the checksums and the parity scheme.
Identifiers of the form `D-0xx` and `H-xx` that appear in source comments refer to design decisions and
experiments recorded in the development notes, which are not part of this repository.

## About this repository

The sources here are exported from the development tree; each commit is a snapshot of it. Please open an
issue for bugs, questions and proposals.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at
your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this project shall be dual licensed as above, without any additional terms or conditions.
