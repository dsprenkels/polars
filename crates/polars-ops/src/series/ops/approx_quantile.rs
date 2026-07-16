use polars_compute::kll::KLLSketch;
use polars_core::prelude::*;

pub fn approx_quantile(s: &Column, quantile: &Series, error: f64) -> PolarsResult<Scalar> {
    let s = s.as_materialized_series();
    let ca: &Float64Chunked = s.as_ref().as_ref();
    let mut sketch = KLLSketch::new(error);
    for item in ca.iter() {
        if let Some(item) = item {
            sketch.update(item);
        }
    }
    dbg!(sketch);
    todo!()
}
