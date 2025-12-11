use byteorder::{ByteOrder, LittleEndian};
use gnuplot::*;
use hnsw::*;
use memmap2::MmapMut;
use rand::distributions::Standard;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use space::Metric;
use space::Neighbor;
use std::cell::RefCell;
use std::fs::OpenOptions;
use std::io::Read;
use std::path::PathBuf;
use structopt::StructOpt;

struct Euclidean;

impl Metric<&[f32]> for Euclidean {
    type Unit = u32;
    fn distance(&self, a: &&[f32], b: &&[f32]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            .to_bits()
    }
}

impl Metric<[f32; 64]> for Euclidean {
    type Unit = u32;
    fn distance(&self, a: &[f32; 64], b: &[f32; 64]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            .to_bits()
    }
}

/// A mmap-based feature store for [f32; 64] arrays.
struct MmapFeatureStore {
    mmap: MmapMut,
    len: usize,
    capacity: usize,
}

impl MmapFeatureStore {
    fn new(path: &str, capacity: usize) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let total_size = capacity * std::mem::size_of::<[f32; 64]>();
        file.set_len(total_size as u64)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            len: 0,
            capacity,
        })
    }
}

impl FeatureStore<[f32; 64]> for MmapFeatureStore {
    fn get(&self, index: usize) -> &[f32; 64] {
        let offset = index * std::mem::size_of::<[f32; 64]>();
        unsafe { &*(self.mmap[offset..].as_ptr() as *const [f32; 64]) }
    }

    fn push(&mut self, feature: [f32; 64]) {
        assert!(self.len < self.capacity, "MmapFeatureStore capacity exceeded");
        let offset = self.len * std::mem::size_of::<[f32; 64]>();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(feature.as_ptr() as *const u8, std::mem::size_of::<[f32; 64]>())
        };
        self.mmap[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Debug, StructOpt)]
#[structopt(name = "recall", about = "Generates recall graphs for HNSW")]
struct Opt {
    /// The value of M to use.
    #[structopt(short = "m", long = "max_edges", default_value = "24")]
    m: usize,
    /// The dataset size to test on.
    #[structopt(short = "s", long = "size", default_value = "10000")]
    size: usize,
    /// Total number of query bitstrings.
    #[structopt(short = "q", long = "queries", default_value = "10000")]
    num_queries: usize,
    /// The number of dimensions in the feature vector.
    #[structopt(short = "l", long = "dimensions", default_value = "64")]
    dimensions: usize,
    /// The beginning ef value.
    #[structopt(short = "b", long = "beginning_ef", default_value = "1")]
    beginning_ef: usize,
    /// The ending ef value.
    #[structopt(short = "e", long = "ending_ef", default_value = "64")]
    ending_ef: usize,
    /// The number of nearest neighbors.
    #[structopt(short = "k", long = "neighbors", default_value = "2")]
    k: usize,
    /// Use the following file to load the search space.
    #[structopt(short = "f", long = "file")]
    file: Option<PathBuf>,
    /// The descriptor stride length in floats.
    #[structopt(short = "d", long = "descriptor_stride", default_value = "64")]
    descriptor_stride: usize,
    /// efConstruction controlls the quality of the graph at build-time.
    #[structopt(short = "c", long = "ef_construction", default_value = "400")]
    ef_construction: usize,
    /// Use mmap-based feature storage instead of in-memory Vec<T>.
    #[structopt(long = "mmap")]
    mmap: bool,
}

fn process<const M: usize, const M0: usize>(opt: &Opt) -> (Vec<f64>, Vec<f64>) {
    assert!(
        opt.k <= opt.size,
        "You must choose a dataset size larger or equal to the test search size"
    );
    let rng = Pcg64::from_seed([5; 32]);

    let (search_space, query_strings): (Vec<f32>, Vec<f32>) = if let Some(filepath) = &opt.file {
        eprintln!(
            "Reading {} search space descriptors of size {} f32s from file \"{}\"...",
            opt.size, opt.descriptor_stride, filepath.display()
        );
        let mut file = std::fs::File::open(filepath).expect("unable to open file");
        let mut search_space = vec![0u8; opt.size * opt.descriptor_stride * 4];
        file.read_exact(&mut search_space).expect(
            "unable to read enough search descriptors from the file (try decreasing -s/-q)",
        );
        let search_space = search_space.chunks_exact(4).map(LittleEndian::read_f32).collect();
        eprintln!("Done.");

        eprintln!(
            "Reading {} query descriptors of size {} f32s from file \"{}\"...",
            opt.num_queries, opt.descriptor_stride, filepath.display()
        );
        let mut query_strings = vec![0u8; opt.num_queries * opt.descriptor_stride * 4];
        file.read_exact(&mut query_strings)
            .expect("unable to read enough query descriptors from the file (try decreasing -q/-s)");
        let query_strings = query_strings.chunks_exact(4).map(LittleEndian::read_f32).collect();
        eprintln!("Done.");

        (search_space, query_strings)
    } else {
        eprintln!("Generating {} random vectors...", opt.size);
        let search_space: Vec<f32> = rng
            .sample_iter(&Standard)
            .take(opt.size * opt.descriptor_stride)
            .collect();
        eprintln!("Done.");

        let rng = Pcg64::from_seed([6; 32]);
        eprintln!("Generating {} independent random query vectors...", opt.num_queries);
        let query_strings: Vec<f32> = rng
            .sample_iter(&Standard)
            .take(opt.num_queries * opt.descriptor_stride)
            .collect();
        eprintln!("Done.");
        (search_space, query_strings)
    };

    let search_space: Vec<_> = search_space
        .chunks_exact(opt.descriptor_stride)
        .map(|c| &c[..opt.dimensions])
        .collect();
    let query_strings: Vec<_> = query_strings
        .chunks_exact(opt.descriptor_stride)
        .map(|c| &c[..opt.dimensions])
        .collect();

    eprintln!("Computing the correct nearest neighbor distance for all {} queries...", opt.num_queries);
    let correct_worst_distances: Vec<_> = query_strings
        .iter()
        .cloned()
        .map(|feature| {
            let mut v = vec![];
            for distance in search_space.iter().map(|n| Euclidean.distance(n, &feature)) {
                let pos = v.binary_search(&distance).unwrap_or_else(|e| e);
                v.insert(pos, distance);
                if v.len() > opt.k {
                    v.resize_with(opt.k, || unreachable!());
                }
            }
            v.into_iter().take(opt.k).last().unwrap()
        })
        .collect();
    eprintln!("Done.");

    eprintln!("Generating HNSW...");
    let mut hnsw: Hnsw<_, _, Pcg64, M, M0> = Hnsw::new_params(
        Euclidean,
        Params::new().ef_construction(opt.ef_construction),
    );
    let mut searcher: Searcher<_> = Searcher::default();
    for feature in &search_space {
        hnsw.insert(*feature, &mut searcher);
    }
    eprintln!("Done.");

    eprintln!("Computing recall graph...");
    let efs = opt.beginning_ef..=opt.ending_ef;
    let state = RefCell::new((searcher, query_strings.iter().cloned().enumerate().cycle()));
    let (recalls, times): (Vec<f64>, Vec<f64>) = efs
        .map(|ef| {
            let correct = RefCell::new(0usize);
            let dest = vec![Neighbor { index: !0, distance: !0 }; opt.k];
            let stats = easybench::bench_env(dest, |mut dest| {
                let mut refmut = state.borrow_mut();
                let (searcher, query) = &mut *refmut;
                let (ix, query_feature) = query.next().unwrap();
                let correct_worst_distance = correct_worst_distances[ix];
                for &mut neighbor in hnsw.nearest(&query_feature, ef, searcher, &mut dest) {
                    if Euclidean.distance(&search_space[neighbor.index], &query_feature)
                        <= correct_worst_distance
                    {
                        *correct.borrow_mut() += 1;
                    }
                }
            });
            (stats, correct.into_inner())
        })
        .fold((vec![], vec![]), |(mut recalls, mut times), (stats, correct)| {
            times.push((stats.ns_per_iter * 0.1f64.powi(9)).recip());
            recalls.push(correct as f64 / (stats.iterations * opt.k) as f64);
            (recalls, times)
        });
    eprintln!("Done.");

    (recalls, times)
}

fn process_mmap<S: FeatureStore<[f32; 64]>, const M: usize, const M0: usize>(
    opt: &Opt,
    storage: S,
) -> (Vec<f64>, Vec<f64>) {
    assert!(opt.k <= opt.size, "You must choose a dataset size larger or equal to the test search size");
    assert!(opt.dimensions == 64, "Mmap mode only supports 64 dimensions (use -l 64)");
    assert!(opt.file.is_none(), "Mmap mode does not support file input");

    let rng = Pcg64::from_seed([5; 32]);

    eprintln!("Generating {} random vectors...", opt.size);
    let raw: Vec<f32> = rng.clone().sample_iter(&Standard).take(opt.size * 64).collect();
    let search_space: Vec<[f32; 64]> = raw
        .chunks_exact(64)
        .map(|chunk| {
            let mut arr = [0.0f32; 64];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect();
    eprintln!("Done.");

    let rng = Pcg64::from_seed([6; 32]);
    eprintln!("Generating {} independent random query vectors...", opt.num_queries);
    let raw: Vec<f32> = rng.sample_iter(&Standard).take(opt.num_queries * 64).collect();
    let query_strings: Vec<[f32; 64]> = raw
        .chunks_exact(64)
        .map(|chunk| {
            let mut arr = [0.0f32; 64];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect();
    eprintln!("Done.");

    eprintln!("Computing the correct nearest neighbor distance for all {} queries...", opt.num_queries);
    let correct_worst_distances: Vec<_> = query_strings
        .iter()
        .map(|feature| {
            let mut v = vec![];
            for distance in search_space.iter().map(|n| Euclidean.distance(n, feature)) {
                let pos = v.binary_search(&distance).unwrap_or_else(|e| e);
                v.insert(pos, distance);
                if v.len() > opt.k {
                    v.resize_with(opt.k, || unreachable!());
                }
            }
            v.into_iter().take(opt.k).last().unwrap()
        })
        .collect();
    eprintln!("Done.");

    eprintln!("Generating HNSW with MmapFeatureStore...");
    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let mut hnsw: Hnsw<Euclidean, [f32; 64], Pcg64, M, M0, S> =
        Hnsw::new_with_storage_and_params(Euclidean, storage, Params::new().ef_construction(opt.ef_construction), prng);
    let mut searcher: Searcher<_> = Searcher::default();
    for feature in &search_space {
        hnsw.insert(*feature, &mut searcher);
    }
    eprintln!("Done.");

    eprintln!("Computing recall graph...");
    let efs = opt.beginning_ef..=opt.ending_ef;
    let state = RefCell::new((searcher, query_strings.iter().cloned().enumerate().cycle()));
    let (recalls, times): (Vec<f64>, Vec<f64>) = efs
        .map(|ef| {
            let correct = RefCell::new(0usize);
            let dest = vec![Neighbor { index: !0, distance: !0 }; opt.k];
            let stats = easybench::bench_env(dest, |mut dest| {
                let mut refmut = state.borrow_mut();
                let (searcher, query) = &mut *refmut;
                let (ix, query_feature) = query.next().unwrap();
                let correct_worst_distance = correct_worst_distances[ix];
                for &mut neighbor in hnsw.nearest(&query_feature, ef, searcher, &mut dest) {
                    if Euclidean.distance(&search_space[neighbor.index], &query_feature)
                        <= correct_worst_distance
                    {
                        *correct.borrow_mut() += 1;
                    }
                }
            });
            (stats, correct.into_inner())
        })
        .fold((vec![], vec![]), |(mut recalls, mut times), (stats, correct)| {
            times.push((stats.ns_per_iter * 0.1f64.powi(9)).recip());
            recalls.push(correct as f64 / (stats.iterations * opt.k) as f64);
            (recalls, times)
        });
    eprintln!("Done.");

    (recalls, times)
}

macro_rules! process_m {
    ($opt:expr, $m:expr, $m0:expr) => {
        process::<$m, $m0>(&$opt)
    };
}

macro_rules! process_mmap_m {
    ($opt:expr, $m:expr, $m0:expr) => {
        process_mmap::<_, $m, $m0>(&$opt, MmapFeatureStore::new("/tmp/recall_mmap.bin", $opt.size).unwrap())
    };
}

fn main() {
    let opt = Opt::from_args();

    let (recalls, times, storage_type) = if opt.mmap {
        let (r, t) = match opt.m {
            4 => process_mmap_m!(opt, 4, 8),
            8 => process_mmap_m!(opt, 8, 16),
            12 => process_mmap_m!(opt, 12, 24),
            16 => process_mmap_m!(opt, 16, 32),
            20 => process_mmap_m!(opt, 20, 40),
            24 => process_mmap_m!(opt, 24, 48),
            28 => process_mmap_m!(opt, 28, 56),
            32 => process_mmap_m!(opt, 32, 64),
            36 => process_mmap_m!(opt, 36, 72),
            40 => process_mmap_m!(opt, 40, 80),
            44 => process_mmap_m!(opt, 44, 88),
            48 => process_mmap_m!(opt, 48, 96),
            52 => process_mmap_m!(opt, 52, 104),
            _ => {
                eprintln!("Only M between 4 and 52 inclusive and multiples of 4 are allowed");
                return;
            }
        };
        (r, t, "MmapFeatureStore")
    } else {
        let (r, t) = match opt.m {
            4 => process_m!(opt, 4, 8),
            8 => process_m!(opt, 8, 16),
            12 => process_m!(opt, 12, 24),
            16 => process_m!(opt, 16, 32),
            20 => process_m!(opt, 20, 40),
            24 => process_m!(opt, 24, 48),
            28 => process_m!(opt, 28, 56),
            32 => process_m!(opt, 32, 64),
            36 => process_m!(opt, 36, 72),
            40 => process_m!(opt, 40, 80),
            44 => process_m!(opt, 44, 88),
            48 => process_m!(opt, 48, 96),
            52 => process_m!(opt, 52, 104),
            _ => {
                eprintln!("Only M between 4 and 52 inclusive and multiples of 4 are allowed");
                return;
            }
        };
        (r, t, "Vec<T>")
    };

    let mut fg = Figure::new();
    fg.axes2d()
        .set_title(
            &format!(
                "{}-NN Recall Graph (dimensions = {}, size = {}, M = {}, storage = {})",
                opt.k, opt.dimensions, opt.size, opt.m, storage_type
            ),
            &[],
        )
        .set_x_label("Recall Rate", &[])
        .set_y_label("Lookups per second", &[])
        .lines(&recalls, &times, &[LineWidth(2.0), Color("blue")])
        .set_y_ticks(Some((Auto, 2)), &[], &[])
        .set_grid_options(true, &[LineStyle(DotDotDash), Color("black")])
        .set_minor_grid_options(&[LineStyle(SmallDot), Color("red")])
        .set_x_grid(true)
        .set_y_grid(true)
        .set_y_minor_grid(true);

    fg.show().ok();
}
