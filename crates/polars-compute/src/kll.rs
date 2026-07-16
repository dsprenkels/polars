use std::fmt;

use polars_utils::scratch_vec::ScratchVec;
use polars_utils::total_ord::TotalOrd;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// KLL calls this `δ`. Equivalent to 99% success rate.
const FAILURE_PROBABILITY: f64 = 0.01;
/// `CAPACITY_DECAY` specifies how much smaller compactor h+1 is wrt to h.
/// KLL calls this `c`.
const CAPACITY_DECAY: f64 = 2.0 / 3.0;

fn compute_k(error: f64, delta: f64, c: f64) -> usize {
    /*
    This is still claude output. [amber] Please check:

    Smallest `k` guaranteeing rank error <= `error * n` w.p. >= 1 - `delta`.

    This is the sizing for the PLAIN varying-capacity hierarchy (this class).
    The sampler variant has its own `SamplingKLLSketch.k_for_error`.

    The randomized compactions make the estimated rank of any fixed value an
    unbiased random walk whose variance is a geometric series over the levels:

        Var(rank error) ~ n**2 * (c / (2c - 1)) / k**2,

    so the error std, as a fraction of the stream length, is ~sqrt(c/(2c-1))/k.
    The error concentrates like a Gaussian, so it stays below `z * eps_std`
    except with probability `delta`, `z = sqrt(2 ln(2/delta))`:

        k = z * sqrt(c / (2c - 1)) / error.

    `heuristic_result` below is that (slightly loose) form; `paper_result` is
    KLL Theorem 1 exactly, `k = (1/error) sqrt(log(2/delta) / C)`, `C=c^2(2c-1)`.
    `c` must exceed 1/2, else the variance series diverges.
    */

    let z = f64::sqrt(2.0 * f64::ln(2.0 / delta)); // Gaussian tail factor for prob. 1 - delta
    let spread = f64::sqrt(c / (2.0 * c - 1.0)); // sqrt of the per-level variance series
    let heuristic_result = f64::max(2.0, f64::ceil(z * spread / error)) as usize;
    heuristic_result
}

#[derive(Debug, Clone, Copy)]
struct Level {
    offset: usize,
    size: usize,
}

#[derive(Debug)]
pub struct KLLSketch<T: fmt::Debug + Clone + TotalOrd> {
    /// Contents of the compactors. The offsets of the compactors are stored
    /// in the levels vector. The top-level compactor is stored at the start
    /// of this Vec, and the bottom-most compactor is stored at the end of this
    /// Vec.
    ///
    /// This algorithm uses the convention that the top-level compactor has
    /// *level* h-1.  The bottom-level compactor has *level* h,
    /// and height *0*. So the order of `levels` is *reversed* wrt `items`.
    items: Vec<T>,
    levels: Vec<Level>,
    /// Unsorted ingestation buffer of all of the items.
    consumed_items: usize,
    max_items: usize,
    k: usize,
    rng: SmallRng,
    scratch: ScratchVec<T>,
}

impl<T: fmt::Debug + Clone + TotalOrd> KLLSketch<T> {
    pub fn new(error: f64) -> Self {
        let k = compute_k(error, FAILURE_PROBABILITY, CAPACITY_DECAY);
        dbg!(k);

        // The expected capacity of the vec is equal to the sum of the size of each compactor,
        // which is [k, (2/3)*k, (2/3)²*k, ...].  The sum of this geometric series is equal to
        // k / (1 - 2/3) = 3*k
        let max_items = k;
        let items: Vec<T> = Vec::with_capacity(3 * k);

        let topmost_level = Level { offset: 0, size: 0 };
        let levels = vec![topmost_level];
        let rng = SmallRng::from_rng(&mut rand::rng());

        Self {
            items,
            levels,
            consumed_items: 0,
            max_items,
            k,
            rng,
            scratch: ScratchVec::with_capacity(k),
        }
    }

    pub fn update(&mut self, val: T) {
        self.items.push(val);
        self.consumed_items += 1;
        self.levels.last_mut().unwrap().size += 1;
        if self.items.len() > self.max_items {
            self.compact();
        }
    }

    pub fn finalize(&mut self) {
        todo!()
    }

    pub fn estimate_rank(&mut self, _val: T) {
        todo!()
    }

    pub fn estimate_quantile(&mut self, _quantile: f64) -> T {
        todo!()
    }

    fn compact(&mut self) {
        let num_levels = self.levels.len();
        let Some((level, _)) = self
            .levels
            .iter()
            .enumerate()
            .find(|(h, l)| l.size >= compactor_threshold(self.k, num_levels - h - 1))
        else {
            return;
        };

        if level == self.levels.len() - 1 {
            self.add_new_compactor();
        }
        self.compact_level(level);
    }

    fn add_new_compactor(&mut self) {
        // Add new compactor to the end of self.levels
        self.levels.push(Level { offset: 0, size: 0 });
        self.max_items += compactor_threshold(self.k, self.levels.len() - 1);
    }

    fn compact_level(&mut self, level: usize) {
        let mut compact_level = self.levels[level];
        let mut next_level = self.levels[level + 1];
        let compact_start = compact_level.offset;
        let compact_end = compact_start + compact_level.size;
        let next_start = next_level.offset;
        let next_end = next_start + next_level.size;
        let buf = self.scratch.get();

        // The base compactor is not sorted yet
        if level == self.levels.len() - 1 {
            self.items[compact_start..compact_end].sort_unstable_by(TotalOrd::tot_cmp);
        }

        let next_level_items = self.items[next_start..next_end].iter().cloned();
        let mut compacted_items = self.items[compact_start..compact_end].iter().cloned();

        // If there is an odd number of items in this compactor, stash the "straggler" to add it back later
        let mut straggler = None;
        if compact_level.size % 2 == 1 {
            straggler = compacted_items.next_back();
        }

        // Throw away half of the values during the compaction
        let coin: bool = self.rng.random();
        if coin {
            compacted_items.next();
        }
        let compacted_items = compacted_items.step_by(2);

        // Merge the items into the next compactor
        merge_sorted(buf, next_level_items, compacted_items);
        next_level.size = buf.len();
        self.items[next_start..next_start + next_level.size].clone_from_slice(&buf);

        // Add back the straggler
        compact_level.offset = next_level.offset + next_level.size;
        if let Some(item) = straggler {
            self.items[compact_level.offset] = item;
            compact_level.size = 1;
        } else {
            compact_level.size = 0;
        }
        let new_compact_end = compact_level.offset + compact_level.size;

        // Shift all of the compactors below the current one
        let shift = compact_end - new_compact_end;
        self.items.drain(new_compact_end..compact_end);
        for level_below_compact in self.levels[..level].iter_mut() {
            level_below_compact.offset -= shift;
        }
        self.levels[level] = compact_level;
        self.levels[level + 1] = next_level;
    }
}

fn compactor_threshold(k: usize, depth: usize) -> usize {
    let depth = u32::try_from(depth).expect("overflow");
    usize::max(k * 2usize.pow(depth).div_ceil(3usize.pow(depth)), 2)
}

fn merge_sorted<T: TotalOrd>(
    vec: &mut Vec<T>,
    iter1: impl Iterator<Item = T>,
    iter2: impl Iterator<Item = T>,
) {
    let mut iter1 = iter1.peekable();
    let mut iter2 = iter2.peekable();
    loop {
        match (iter1.peek(), iter2.peek()) {
            (None, None) => return,
            (Some(_), None) => vec.push(iter1.next().unwrap()),
            (None, Some(_)) => vec.push(iter2.next().unwrap()),
            (Some(x1), Some(x2)) => {
                if TotalOrd::tot_le(x1, x2) {
                    vec.push(iter1.next().unwrap());
                } else {
                    vec.push(iter2.next().unwrap())
                }
            },
        }
    }
}
