use hnsw::{FeatureStore, Hnsw, Params, Searcher};
use rand::Rng;
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};
use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

struct Euclidean;

impl Metric<[f32; 128]> for Euclidean {
    type Unit = u32;
    fn distance(&self, a: &[f32; 128], b: &[f32; 128]) -> u32 {
        let sum: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        (sum.sqrt() * 1_000_000.0) as u32
    }
}

const FEATURE_SIZE: usize = 128;
const FEATURE_BYTES: usize = FEATURE_SIZE * std::mem::size_of::<f32>();

/// A disk-based feature store that reads features from disk on every access.
/// This uses NO RAM for feature storage - only a small buffer for reading.
///
/// Trade-off: Much slower due to disk I/O, but uses minimal memory. we are not
/// using mmapping here to isolate memory usage as low as possible (mmap is faster but
/// measurements are affected by OS page caching. realistically the results of this storage impl
/// show the true memory usage).
struct DiskFeatureStore {
    file: File,
    len: usize,
}

impl DiskFeatureStore {
    fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        Ok(Self { file, len: 0 })
    }
}

impl FeatureStore<[f32; 128]> for DiskFeatureStore {
    fn get(&self, index: usize) -> &[f32; 128] {
        // Use thread-local storage to return a reference
        // This is the key trick - we only keep ONE feature in memory at a time
        thread_local! {
            static BUFFER: std::cell::UnsafeCell<[f32; 128]> = std::cell::UnsafeCell::new([0.0; 128]);
        }

        BUFFER.with(|buf| {
            let buffer = unsafe { &mut *buf.get() };

            let mut file = &self.file;
            let offset = index * FEATURE_BYTES;
            file.seek(SeekFrom::Start(offset as u64)).unwrap();

            let mut bytes = [0u8; FEATURE_BYTES];
            file.read_exact(&mut bytes).unwrap();

            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                buffer[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }

            unsafe { &*(buffer as *const [f32; 128]) }
        })
    }

    fn push(&mut self, feature: [f32; 128]) {
        self.file.seek(SeekFrom::End(0)).unwrap();
        let bytes: Vec<u8> = feature.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.file.write_all(&bytes).unwrap();
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A "null" feature store that doesn't store features at all.
/// Only useful for measuring the memory overhead of the graph structure alone.
struct NullFeatureStore {
    len: usize,
    dummy: [f32; 128],
}

impl NullFeatureStore {
    fn new() -> Self {
        Self {
            len: 0,
            dummy: [0.0; 128],
        }
    }
}

impl FeatureStore<[f32; 128]> for NullFeatureStore {
    fn get(&self, _index: usize) -> &[f32; 128] {
        // WARNING: This returns garbage! Only for memory measurement.
        &self.dummy
    }

    fn push(&mut self, _feature: [f32; 128]) {
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

fn generate_random_vectors(count: usize) -> Vec<[f32; 128]> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let mut v = [0.0f32; 128];
            for x in v.iter_mut() {
                *x = rng.gen();
            }
            v
        })
        .collect()
}

fn get_memory_usage() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            let parts: Vec<&str> = statm.split_whitespace().collect();
            if parts.len() >= 2 {
                let rss_pages: usize = parts[1].parse().unwrap_or(0);
                return rss_pages * 4096;
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
        {
            if let Ok(rss_str) = String::from_utf8(output.stdout) {
                if let Ok(rss_kb) = rss_str.trim().parse::<usize>() {
                    return rss_kb * 1024;
                }
            }
        }
    }

    0
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

fn main() -> std::io::Result<()> {
    const NUM_VECTORS: usize = 100_000;
    const NUM_QUERIES: usize = 10;

    let params = Params::new().ef_construction(20); // Lower for faster insertion

    // Generate test data
    println!("Generating random vectors...");
    let vectors = generate_random_vectors(NUM_VECTORS);
    let queries = generate_random_vectors(NUM_QUERIES);

    // Drop the generated vectors from consideration
    let vectors_size = vectors.len() * FEATURE_BYTES;

    println!("=== Test 1: Graph structure only (NullFeatureStore) ===");

    let mem_before_null = get_memory_usage();
    let graph_only_delta;
    {
        let storage = NullFeatureStore::new();
        let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

        let mut hnsw: Hnsw<Euclidean, [f32; 128], Pcg64, 12, 24, NullFeatureStore> =
            Hnsw::new_with_storage_and_params(Euclidean, storage, params.clone(), prng);
        let mut searcher = Searcher::default();

        let start = Instant::now();
        for v in &vectors {
            hnsw.insert(*v, &mut searcher);
        }

        let mem_after = get_memory_usage();
        graph_only_delta = mem_after.saturating_sub(mem_before_null);
        println!("Graph-only memory: +{}", format_bytes(graph_only_delta));
    }
    println!();

    println!("=== Test 2: In-memory Vec<T> (default) ===");

    let mem_before_vec = get_memory_usage();
    let vec_delta;
    {
        let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
        let mut hnsw: Hnsw<Euclidean, [f32; 128], Pcg64, 12, 24> =
            Hnsw::new_params_and_prng(Euclidean, params.clone(), prng);
        let mut searcher = Searcher::default();

        let start = Instant::now();
        for v in &vectors {
            hnsw.insert(*v, &mut searcher);
        }

        let mem_after = get_memory_usage();
        vec_delta = mem_after.saturating_sub(mem_before_vec);
        println!("Total memory: +{}", format_bytes(vec_delta));

        let mut neighbors = [Neighbor { index: !0, distance: !0 }; 10];
        let start = Instant::now();
        for q in &queries {
            hnsw.nearest(q, 64, &mut searcher, &mut neighbors);
        }
        println!("Search time: {:?} ({} queries)", start.elapsed(), NUM_QUERIES);
    }
    println!();

    println!("=== Test 3: Disk-based FeatureStore ===");

    let disk_path = "/tmp/hnsw_disk_features.bin";
    let mem_before_disk = get_memory_usage();
    let disk_delta;
    {
        let storage = DiskFeatureStore::new(disk_path)?;
        let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

        let mut hnsw: Hnsw<Euclidean, [f32; 128], Pcg64, 12, 24, DiskFeatureStore> =
            Hnsw::new_with_storage_and_params(Euclidean, storage, params.clone(), prng);
        let mut searcher = Searcher::default();

        let start = Instant::now();
        for v in &vectors {
            hnsw.insert(*v, &mut searcher);
        }

        let mem_after = get_memory_usage();
        disk_delta = mem_after.saturating_sub(mem_before_disk);

        // Search (will be slow due to disk I/O)
        let mut neighbors = [Neighbor { index: !0, distance: !0 }; 10];
        let start = Instant::now();
        for q in &queries {
            hnsw.nearest(q, 64, &mut searcher, &mut neighbors);
        }
        println!("Search time: {:?} ({} queries) - slower due to disk I/O", start.elapsed(), NUM_QUERIES);

        std::fs::remove_file(disk_path).ok();
    }
    println!();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      MEMORY SUMMARY                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Feature data size:          {:>30} ║", format_bytes(NUM_VECTORS * FEATURE_BYTES));
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Graph structure only:       {:>30} ║", format_bytes(graph_only_delta));
    println!("║ Vec<T> (graph + features):  {:>30} ║", format_bytes(vec_delta));
    println!("║ Disk-based (graph only):    {:>30} ║", format_bytes(disk_delta));
    println!("╠══════════════════════════════════════════════════════════════╣");

    let feature_overhead = vec_delta.saturating_sub(graph_only_delta);
    let savings = vec_delta.saturating_sub(disk_delta);
    let savings_pct = if vec_delta > 0 {
        (savings as f64 / vec_delta as f64) * 100.0
    } else {
        0.0
    };

    println!("║ Feature storage overhead:   {:>30} ║", format_bytes(feature_overhead));
    println!("║ Memory saved with disk:     {:>30} ║", format_bytes(savings));
    println!("║ Savings percentage:         {:>29.1}% ║", savings_pct);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}
