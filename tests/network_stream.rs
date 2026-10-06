//! Real network-socket verification for streaming.
//!
//! `encoder::StreamEncoder`/`decoder::decode_stream` are generic over `std::io::Write`/
//! `std::io::Read`, and's "Streaming" section claims this means the
//! format works over "a real pipe or socket". But only ever actually exercised a
//! pipe (`fak stream-encode | fak stream-decode`) -- never a real network socket, which is a
//! different code path on every OS (different buffering, different partial-read/partial-write
//! behavior). SS40: don't claim a socket works without having run it over one. This test
//! closes that gap with a genuine TCP loopback connection: one thread is the "network sender"
//! (`StreamEncoder` writing straight to a `TcpStream`), the main thread is the "network receiver"
//! (`decode_stream` reading from the peer `TcpStream` returned by a real `accept()`).

use fak::decoder::decode_stream;
use fak::encoder::StreamEncoder;
use fak::format::MODE_BLOCK_INDEPENDENT;
use fak::metadata::Metadata;
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn signal(n: usize, seed: u64) -> Vec<i64> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((i as f64 * 0.013).sin() * 9000.0) as i64 + (s % 301) as i64 - 150
        })
        .collect()
}

/// Streams 6 chunks of real stereo audio over an actual TCP loopback connection with an
/// artificial ~40ms gap between pushes (simulating a live network source arriving over time, not
/// a file dumped all at once), and confirms: (1) the decoded PCM is bit-exact against what was
/// pushed, over the real socket; (2) the receiver's `on_chunk` callback fires progressively as
/// each chunk arrives on the wire -- not all at once after the sender finishes -- which is the
/// actual thing "streamable over the network" means (a peer can start playing before the whole
/// stream has arrived), not merely that the bytes are transport-agnostic.
#[test]
fn stream_encoder_decoder_round_trip_over_real_tcp_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let chunks: Vec<Vec<Vec<i64>>> = (0..6)
        .map(|i| {
            vec![
                signal(3000 + i * 91, 100 + i as u64),
                signal(3000 + i * 91, 900 + i as u64),
            ]
        })
        .collect();
    let full: Vec<Vec<i64>> = vec![
        chunks.iter().flat_map(|c| c[0].clone()).collect(),
        chunks.iter().flat_map(|c| c[1].clone()).collect(),
    ];
    let n_pushed = chunks.len();

    let (arrival_tx, arrival_rx) = mpsc::channel::<Instant>();

    let sender_chunks = chunks.clone();
    let sender = thread::spawn(move || {
        let stream = TcpStream::connect(addr).unwrap();
        let mut enc =
            StreamEncoder::new(stream, 2, 44100, 16, MODE_BLOCK_INDEPENDENT, &Metadata::default())
                .unwrap();
        for c in &sender_chunks {
            enc.push_chunk(c).unwrap();
            thread::sleep(Duration::from_millis(40));
        }
        enc.finish().unwrap();
    });

    let (socket, _) = listener.accept().unwrap();
    let mut got: Vec<Vec<i64>> = vec![Vec::new(); 2];
    let (header, _meta) = decode_stream(socket, |ch| {
        arrival_tx.send(Instant::now()).unwrap();
        for (dst, src) in got.iter_mut().zip(ch) {
            dst.extend_from_slice(src);
        }
        Ok(())
    })
    .unwrap();
    sender.join().unwrap();

    assert_eq!(header.channels, 2);
    assert_eq!(
        got, full,
        "decoded PCM over the real TCP socket must be bit-exact against what was pushed"
    );

    let arrivals: Vec<Instant> = arrival_rx.try_iter().collect();
    assert_eq!(arrivals.len(), n_pushed);
    // Progressive delivery: if `decode_stream` were secretly buffering the whole connection before
    // decoding anything, every arrival timestamp would cluster within a few ms of each other at the
    // very end. Requiring at least half the sender's total pacing budget to have elapsed between
    // the first and last callback proves the decoder is genuinely consuming and decoding chunks as
    // they land on the wire, not waiting for the peer to close the connection.
    let span = arrivals[arrivals.len() - 1].duration_since(arrivals[0]);
    let sender_budget = Duration::from_millis(40 * (n_pushed as u64 - 1));
    assert!(
        span >= sender_budget / 2,
        "chunks arrived too close together ({span:?} over a {sender_budget:?} sender budget) for \
         genuinely progressive over-the-wire delivery"
    );
}

/// Same real-socket transport, but for an ordinary known-length stream (the common case: encoding
/// a file you already have and pushing it out to a network peer) -- confirms `decode_stream`'s
/// EOF/trailing-garbage handling also behaves correctly when the peer end of the connection is a
/// real socket that gets closed by the OS, not an in-memory slice ending.
#[test]
fn known_length_stream_over_real_tcp_socket_and_clean_close() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let chans = vec![signal(20_000, 7), signal(20_000, 8)];
    // FEC disabled: decode_stream never understands FEC parity blocks (file-based
    // path only) -- a real FEC-enabled file is outside its input domain.
    let encoded = fak::encoder::encode_chunked(
        &chans,
        44100,
        16,
        MODE_BLOCK_INDEPENDENT,
        6000,
        1,
        None,
        &Metadata::default(),
    )
    .unwrap();

    let payload = encoded.clone();
    let sender = thread::spawn(move || {
        use std::io::Write;
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(&payload).unwrap();
        stream.shutdown(std::net::Shutdown::Write).unwrap();
    });

    let (socket, _) = listener.accept().unwrap();
    let mut got: Vec<Vec<i64>> = vec![Vec::new(); 2];
    let (header, _meta) = decode_stream(socket, |ch| {
        for (dst, src) in got.iter_mut().zip(ch) {
            dst.extend_from_slice(src);
        }
        Ok(())
    })
    .unwrap();
    sender.join().unwrap();

    assert_eq!(header.total_frames, 20_000);
    assert_eq!(got, chans);
}
