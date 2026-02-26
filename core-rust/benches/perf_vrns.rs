use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gmem_rs::vrns::scalar_synth::synthesize_multichannel_u64;

fn benchmark_vrns_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("vRNS Synthesis");

    // Test base throughput.
    // baseline in python was ~110,000 ops/sec (approx 9 microseconds per op).
    group.bench_function("synthesize_16_prime_fmix64", |b| {
        let mut addr = 0xBEEFCACE0000;
        let seed = 0x1337BEEF9000CAFE;
        b.iter(|| {
            // We increment addr to prevent the compiler from caching the result entirely.
            // black_box ensures the compiler does not optimize away the loop.
            let res = synthesize_multichannel_u64(black_box(addr), black_box(seed));
            addr = addr.wrapping_add(1);
            black_box(res);
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_vrns_throughput);
criterion_main!(benches);
