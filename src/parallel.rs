//! Minimal scoped work pool for chunk-parallel encode and decode. No dependencies: `std::thread::scope` plus an atomic
//! work index. Results come back in input order, so the encoded bytes never depend on the thread
//! count or on scheduling -- a 1-thread and an N-thread encode of the same input are byte-identical.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Worker count used when the caller doesn't specify one: every hardware thread the OS reports.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// `items.iter().map(f).collect()`, spread over up to `threads` workers. Workers pull the next
/// unclaimed index, so uneven item costs (e.g. a short final chunk) still balance.
pub fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = threads.max(1).min(items.len());
    if threads <= 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<R>>> = (0..items.len()).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() {
                    break;
                }
                let r = f(&items[i]);
                *results[i].lock().unwrap() = Some(r);
            });
        }
    });
    results.into_iter().map(|m| m.into_inner().unwrap().expect("every index is claimed exactly once")).collect()
}

/// Runs `produce(0..n)` on up to `threads` workers and hands each result to `consume` on the
/// calling thread, in index order, as soon as it and every earlier one are ready:
/// decoding and writing overlap, with no barrier between batches and one set of threads for the
/// whole run. Workers stay at most `window` items ahead of `consume`, so at most `window` results
/// wait in memory. The first error from `consume` stops the workers (items already started finish)
/// and is returned. Order, and so the output, never depends on `threads`.
pub fn ordered_pipeline<R: Send, E>(n: usize, threads: usize, window: usize, produce: impl Fn(usize) -> R + Sync,
                                    mut consume: impl FnMut(usize, R) -> Result<(), E>) -> Result<(), E> {
    use std::collections::HashMap;
    use std::sync::Condvar;
    let threads = threads.max(1).min(n);
    if threads <= 1 {
        for i in 0..n { consume(i, produce(i))?; }
        return Ok(());
    }
    let window = window.max(threads);
    struct State<R> { next: usize, consumed: usize, stop: bool, ready: HashMap<usize, R>, panic: Option<Box<dyn std::any::Any + Send>> }
    let state = Mutex::new(State { next: 0, consumed: 0, stop: false, ready: HashMap::new(), panic: None });
    let cv = Condvar::new();
    let outcome = std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = {
                    let mut st = cv.wait_while(state.lock().unwrap(), |st| !st.stop && st.next < n && st.next >= st.consumed + window).unwrap();
                    if st.stop || st.next >= n { break; }
                    st.next += 1;
                    st.next - 1
                };
                // A panicking `produce` must not leave the consumer waiting for its item forever.
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| produce(i))) {
                    Ok(r) => { state.lock().unwrap().ready.insert(i, r); }
                    Err(payload) => {
                        let mut st = state.lock().unwrap();
                        st.panic.get_or_insert(payload);
                        st.stop = true;
                    }
                }
                cv.notify_all();
            });
        }
        let mut result = Ok(());
        for i in 0..n {
            let r = {
                let mut st = cv.wait_while(state.lock().unwrap(), |st| !st.ready.contains_key(&i) && st.panic.is_none()).unwrap();
                if st.panic.is_some() { break; }
                st.consumed = i + 1;
                st.ready.remove(&i).expect("present")
            };
            cv.notify_all();
            if let Err(e) = consume(i, r) { result = Err(e); break; }
        }
        state.lock().unwrap().stop = true;
        cv.notify_all();
        result
    });
    if let Some(payload) = state.into_inner().unwrap().panic { std::panic::resume_unwind(payload); }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_pipeline_delivers_in_order_for_every_thread_count() {
        for threads in [0, 1, 2, 3, 8, 64] {
            for window in [0, 1, 4, 100] {
                let mut got = Vec::new();
                // Uneven item costs, so completions arrive out of order.
                let r: Result<(), ()> = ordered_pipeline(257, threads, window, |i| {
                    std::thread::sleep(std::time::Duration::from_micros(((i * 37) % 11) as u64 * 20));
                    i * i + 1
                }, |i, v| { assert_eq!(v, i * i + 1); got.push(i); Ok(()) });
                assert!(r.is_ok());
                assert_eq!(got, (0..257).collect::<Vec<_>>(), "threads={threads} window={window}");
            }
        }
    }

    #[test]
    fn ordered_pipeline_propagates_a_panicking_item_instead_of_hanging() {
        for threads in [2, 4] {
            let r = std::panic::catch_unwind(|| {
                let _: Result<(), ()> = ordered_pipeline(64, threads, 8, |i| { if i == 20 { panic!("item {i} failed"); } i }, |_, _| Ok(()));
            });
            assert!(r.is_err(), "threads {threads}: the panic must reach the caller");
        }
    }

    #[test]
    fn ordered_pipeline_stops_at_the_first_error_and_bounds_lookahead() {
        use std::sync::atomic::AtomicUsize;
        let produced = AtomicUsize::new(0);
        let r = ordered_pipeline(10_000, 8, 16, |i| { produced.fetch_add(1, Ordering::Relaxed); i },
                                 |i, _| if i == 50 { Err(i) } else { Ok(()) });
        assert_eq!(r, Err(50));
        // Workers never ran more than `window` items past what was consumed.
        assert!(produced.load(Ordering::Relaxed) <= 51 + 16, "produced {}", produced.load(Ordering::Relaxed));
        let r: Result<(), ()> = ordered_pipeline(0, 8, 16, |i| i, |_, _| Ok(()));
        assert!(r.is_ok());
    }

    #[test]
    fn preserves_order_for_every_thread_count() {
        let items: Vec<u64> = (0..257).collect();
        let expected: Vec<u64> = items.iter().map(|x| x * x + 1).collect();
        for threads in [0, 1, 2, 3, 8, 64, 1000] {
            assert_eq!(par_map(&items, threads, |x| x * x + 1), expected, "threads={threads}");
        }
    }

    #[test]
    fn empty_input() {
        let items: Vec<u8> = Vec::new();
        assert!(par_map(&items, 8, |x| *x).is_empty());
    }
}
