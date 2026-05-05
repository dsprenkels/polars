use polars_ops::prelude::*;
use recursive::recursive;

use super::*;

// TODO: keep the old binary executor nearby while comparing this experiment.
pub(crate) struct MergeSorted {
    pub(crate) input_left: Box<dyn Executor>,
    pub(crate) input_right: Box<dyn Executor>,
    pub(crate) key: PlSmallStr,
}

impl Executor for MergeSorted {
    #[recursive]
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run MergeSorted")
            }
        }
        let (left, right) = {
            let mut state2 = state.split();
            state2.branch_idx += 1;
            let (left, right) = POOL.join(
                || self.input_left.execute(state),
                || self.input_right.execute(&mut state2),
            );
            (left?, right?)
        };

        let profile_name = Cow::Borrowed("Merge Sorted");
        state.record(
            || {
                let lhs = left.column(self.key.as_str())?;
                let rhs = right.column(self.key.as_str())?;

                _merge_sorted_dfs(
                    &left,
                    &right,
                    lhs.as_materialized_series(),
                    rhs.as_materialized_series(),
                    true,
                )
            },
            profile_name,
        )
    }
}

pub(crate) struct MergeSortedMany {
    pub(crate) inputs: Vec<Box<dyn Executor>>,
    pub(crate) key: PlSmallStr,
}

impl Executor for MergeSortedMany {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run MergeSortedMany")
            }
        }

        // TODO: this is intentionally a little direct for now: gather all inputs,
        // then let the heap/DataFrameBuilder experiment own the full merge.
        let mut inputs = std::mem::take(&mut self.inputs);
        let dfs = if inputs.len() == 1 {
            vec![inputs.pop().unwrap().execute(state)?]
        } else {
            POOL.install(|| {
                inputs
                    .chunks_mut(POOL.current_num_threads() * 3)
                    .map(|chunk| {
                        chunk
                            .into_par_iter()
                            .enumerate()
                            .map(|(idx, input)| {
                                let mut input = std::mem::take(input);
                                let mut state = state.split();
                                state.branch_idx += idx;
                                input.execute(&mut state)
                            })
                            .collect::<PolarsResult<Vec<_>>>()
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?
            .into_iter()
            .flatten()
            .collect()
        };

        let profile_name = Cow::Borrowed("Merge Sorted Many");
        state.record(
            || _merge_sorted_dfs_many(&dfs, self.key.as_str(), true),
            profile_name,
        )
    }
}
