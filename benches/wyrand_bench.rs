use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wyrand::WyRand;

const N: usize = 100_000;

fn bench_f32_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_uniform");
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_f32();
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_f32(&mut buf);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f32_gaussian(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_gaussian");
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_gaussian_f32();
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_gaussian_f32(&mut buf);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f32_symmetric_uncertainty(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_symmetric_uncertainty");
    let mode = 5.0;
    let sigma = 2.0;
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_symmetric_uncertainty_f32(mode, sigma);
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_symmetric_uncertainty_f32(&mut buf, mode, sigma);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f32_asymmetric_uncertainty(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_asymmetric_uncertainty");
    let mode = 5.0;
    let sigma_lo = 1.0;
    let sigma_hi = 3.0;
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_asymmetric_uncertainty_f32(mode, sigma_lo, sigma_hi);
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_asymmetric_uncertainty_f32(&mut buf, mode, sigma_lo, sigma_hi);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f32_rayleigh(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_rayleigh");
    let sigma = 2.0;
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_rayleigh_f32(sigma);
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_rayleigh_f32(&mut buf, sigma);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f64_gaussian(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_gaussian");
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_gaussian_f64();
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_gaussian_f64(&mut buf);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f32_beta(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_beta");
    let alpha = 2.0;
    let beta = 2.0;
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_beta_f32(alpha, beta);
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_beta_f32(&mut buf, alpha, beta);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f64_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_uniform");
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_f64();
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_f64(&mut buf);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f64_rayleigh(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_rayleigh");
    let sigma = 2.0;
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_rayleigh_f64(sigma);
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_rayleigh_f64(&mut buf, sigma);
            black_box(&buf);
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_f32_uniform,
    bench_f32_gaussian,
    bench_f32_symmetric_uncertainty,
    bench_f32_asymmetric_uncertainty,
    bench_f32_rayleigh,
    bench_f64_uniform,
    bench_f64_gaussian,
    bench_f64_rayleigh,
    bench_f32_beta
);
criterion_main!(benches);
