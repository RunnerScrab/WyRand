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
                *val = rng.next_uniform_f32();
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_uniform_f32(&mut buf);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f32_std_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_std_normal");
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_std_normal_f32();
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_std_normal_f32(&mut buf);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f32_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_normal");
    let mode = 5.0;
    let sigma = 2.0;
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_normal_f32(mode, sigma);
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_normal_f32(&mut buf, mode, sigma);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f32_split_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_split_normal");
    let mode = 5.0;
    let sigma_lo = 1.0;
    let sigma_hi = 3.0;
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_split_normal_f32(mode, sigma_lo, sigma_hi);
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_split_normal_f32(&mut buf, mode, sigma_lo, sigma_hi);
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

fn bench_f64_std_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_std_normal");
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_std_normal_f64();
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_std_normal_f64(&mut buf);
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

fn bench_f32_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_poisson");
    let lambda = 5.0;
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0u32; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_poisson_u32(lambda);
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0u32; N];
        b.iter(|| {
            rng.fill_poisson_u32(&mut buf, lambda);
            black_box(&buf);
        })
    });
    group.bench_function("collecting_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0u32; N];
        b.iter(|| {
            rng.fill_poisson_collecting_u32(&mut buf, lambda);
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
                *val = rng.next_uniform_f64();
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_uniform_f64(&mut buf);
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

fn bench_f32_rayleigh_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_rayleigh_cols");
    let sigmas = vec![2.0; N];
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_rayleigh_f32(&mut buf, &sigmas);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f32_normal_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_normal_cols");
    let modes = vec![5.0; N];
    let sigmas = vec![2.0; N];
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0.0; N];
        b.iter(|| {
            rng.fill_normal_f32(&mut buf, &modes, &sigmas);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_f64_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_poisson");
    let lambda = 5.0;
    group.bench_function("scalar", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0u32; N];
        b.iter(|| {
            for val in buf.iter_mut() {
                *val = rng.next_poisson_f64_u32(lambda);
            }
            black_box(&buf);
        })
    });
    group.bench_function("bulk_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0u32; N];
        b.iter(|| {
            rng.fill_poisson_f64_u32(&mut buf, lambda);
            black_box(&buf);
        })
    });
    group.bench_function("collecting_api", |b| {
        let mut rng = WyRand::new(1);
        let mut buf = vec![0u32; N];
        b.iter(|| {
            rng.fill_poisson_collecting_f64_u32(&mut buf, lambda);
            black_box(&buf);
        })
    });
    group.finish();
}

fn bench_radian_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("radian_sampling");
    group.bench_function("f32_on_sphere", |b| {
        let mut rng = WyRand::new(1);
        b.iter(|| {
            black_box(rng.next_isotropic_polar_angle_f32());
        })
    });
    group.bench_function("f64_on_sphere", |b| {
        let mut rng = WyRand::new(1);
        b.iter(|| {
            black_box(rng.next_isotropic_polar_angle_f64());
        })
    });
    group.finish();
}


criterion_group!(
    benches,
    bench_f32_uniform,
    bench_f32_std_normal,
    bench_f32_normal,
    bench_f32_split_normal,
    bench_f32_rayleigh,
    bench_f64_uniform,
    bench_f64_std_normal,
    bench_f64_rayleigh,
    bench_f32_beta,
    bench_f32_rayleigh_cols,
    bench_f32_normal_cols,
    bench_f32_poisson,
    bench_f64_poisson,
    bench_radian_sampling
);
criterion_main!(benches);
