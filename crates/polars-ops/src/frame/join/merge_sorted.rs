use std::cmp::Ordering;
use std::collections::BinaryHeap;

use arrow::legacy::utils::CustomIterTools;
#[cfg(feature = "dtype-categorical")]
use polars_core::datatypes::CategoricalPhysical;
use polars_core::prelude::*;
#[cfg(feature = "dtype-categorical")]
use polars_core::with_match_categorical_physical_type;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::ToTotalOrd;

#[derive(Clone, Copy, Eq, PartialEq)]
struct HeapEntry<K> {
    input_idx: usize,
    key: K,
}

impl<K: Ord> Ord for HeapEntry<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .cmp(&self.key)
            .then_with(|| other.input_idx.cmp(&self.input_idx))
    }
}

impl<K: Ord> PartialOrd for HeapEntry<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn _merge_sorted_dfs_many(
    dfs: &[DataFrame],
    key: &str,
    check_schema: bool,
) -> PolarsResult<DataFrame> {
    let Some((first, rest)) = dfs.split_first() else {
        polars_bail!(NoData: "empty container given");
    };

    if check_schema {
        for df in rest {
            first.schema_equal(df)?;
        }
    }

    let dtype = first.column(key)?.as_materialized_series().dtype().clone();
    let mut non_empty = Vec::with_capacity(dfs.len());
    for df in dfs {
        let key_s = df.column(key)?.as_materialized_series();
        polars_ensure!(
            key_s.dtype() == &dtype,
            ComputeError: "merge-sort datatype mismatch: {} != {}", dtype, key_s.dtype()
        );

        if !key_s.is_empty() {
            non_empty.push((df, key_s.clone()));
        }
    }

    match non_empty.len() {
        0 => return Ok(first.clone()),
        1 => return Ok(non_empty[0].0.clone()),
        _ => {},
    }

    if non_empty.iter().any(|(_, key_s)| key_s.null_count() > 0) {
        return merge_sorted_dfs_pairwise(dfs, key, check_schema);
    }

    let physical_keys = non_empty
        .iter()
        .map(|(_, key_s)| key_s.to_physical_repr().into_owned())
        .collect::<Vec<_>>();
    let frames = non_empty.iter().map(|(df, _)| *df).collect::<Vec<_>>();

    match physical_keys[0].dtype() {
        dt if dt.is_primitive_numeric() => {
            // TODO: real experiment path. Everything else falls back to the old
            // pairwise kernel until we know this is worth making generic.
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let keys = physical_keys
                    .iter()
                    .map(|s| {
                        let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                        ca
                    })
                    .collect::<Vec<_>>();

                merge_ordered_frames(&frames, |input_idx, idx| unsafe {
                    keys[input_idx].value_unchecked(idx).to_total_ord()
                })
            })
        },
        _ => merge_sorted_dfs_pairwise(dfs, key, check_schema),
    }
}

fn merge_sorted_dfs_pairwise(
    dfs: &[DataFrame],
    key: &str,
    check_schema: bool,
) -> PolarsResult<DataFrame> {
    let Some((first, rest)) = dfs.split_first() else {
        polars_bail!(NoData: "empty container given");
    };

    let mut out = first.clone();
    for df in rest {
        let left_s = out.column(key)?.as_materialized_series();
        let right_s = df.column(key)?.as_materialized_series();
        out = _merge_sorted_dfs(&out, df, left_s, right_s, check_schema)?;
    }
    Ok(out)
}

fn merge_ordered_frames<K: Ord + Copy>(
    dfs: &[&DataFrame],
    key_at: impl Fn(usize, usize) -> K,
) -> PolarsResult<DataFrame> {
    let lengths = dfs.iter().map(|df| df.height()).collect::<Vec<_>>();
    let total_len: usize = lengths.iter().sum();
    let mut merge_indicator: Vec<u8> = Vec::with_capacity(total_len);

    let mut positions = vec![0usize; dfs.len()];
    let mut heap = BinaryHeap::with_capacity(dfs.len());

    for input_idx in 0..dfs.len() {
        heap.push(HeapEntry {
            input_idx,
            key: key_at(input_idx, 0),
        });
    }

    while let Some(HeapEntry { input_idx, .. }) = heap.pop() {
        merge_indicator.push(input_idx as u8);
        positions[input_idx] += 1;

        while positions[input_idx] < lengths[input_idx] {
            let current = key_at(input_idx, positions[input_idx]);
            if heap.peek().is_some_and(|next| current > next.key) {
                break;
            }
            merge_indicator.push(input_idx as u8);
            positions[input_idx] += 1;
        }

        if positions[input_idx] < lengths[input_idx] {
            heap.push(HeapEntry {
                input_idx,
                key: key_at(input_idx, positions[input_idx]),
            });
        }
    }

    let new_columns = (0..dfs[0].width())
        .map(|col_idx| {
            let orig_col = &dfs[0].columns()[col_idx];
            let phys_ss: Vec<Series> = dfs
                .iter()
                .map(|df| {
                    df.columns()[col_idx]
                        .to_physical_repr()
                        .as_materialized_series()
                        .clone()
                })
                .collect();
            let merged = merge_series(&phys_ss, &merge_indicator)?;
            let out = Column::from(merged);
            let mut out = unsafe { out.from_physical_unchecked(orig_col.dtype()) }.unwrap();
            out.rename(orig_col.name().clone());
            Ok(out)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(unsafe { DataFrame::new_unchecked(total_len, new_columns) })
}
pub fn _merge_sorted_dfs(
    left: &DataFrame,
    right: &DataFrame,
    left_s: &Series,
    right_s: &Series,
    check_schema: bool,
) -> PolarsResult<DataFrame> {
    if check_schema {
        left.schema_equal(right)?;
    }
    let dtype_lhs = left_s.dtype();
    let dtype_rhs = right_s.dtype();

    polars_ensure!(
        dtype_lhs == dtype_rhs,
        ComputeError: "merge-sort datatype mismatch: {} != {}", dtype_lhs, dtype_rhs
    );

    // If one frame is empty, we can return the other immediately.
    if right_s.is_empty() {
        return Ok(left.clone());
    } else if left_s.is_empty() {
        return Ok(right.clone());
    }

    let merge_indicator = series_to_merge_indicator(left_s, right_s)?;
    let new_columns = left
        .columns()
        .iter()
        .zip(right.columns())
        .map(|(lhs, rhs)| {
            let lhs_phys = lhs.to_physical_repr();
            let rhs_phys = rhs.to_physical_repr();

            let out = Column::from(merge_series(
                &[
                    lhs_phys.as_materialized_series().clone(),
                    rhs_phys.as_materialized_series().clone(),
                ],
                &merge_indicator,
            )?);

            let mut out = unsafe { out.from_physical_unchecked(lhs.dtype()) }.unwrap();
            out.rename(lhs.name().clone());
            Ok(out)
        })
        .collect::<PolarsResult<_>>()?;

    Ok(unsafe { DataFrame::new_unchecked(left.height() + right.height(), new_columns) })
}

fn merge_series(ss: &[Series], merge_indicator: &[u8]) -> PolarsResult<Series> {
    use DataType::*;
    let out = match ss[0].dtype() {
        Null => Series::new_null(PlSmallStr::EMPTY, merge_indicator.len()),
        Boolean => {
            let ss = ss.iter().map(|s| s.bool().unwrap()).collect_vec();
            merge_ca(&ss[..], merge_indicator).into_series()
        },
        String => {
            // dispatch via binary
            let binaries = ss
                .iter()
                .map(|s| s.str().unwrap().as_binary())
                .collect_vec();
            let refs: Vec<&BinaryChunked> = binaries.iter().collect();
            let out = merge_ca(&refs, merge_indicator);
            unsafe { out.to_string_unchecked() }.into_series()
        },
        Binary => {
            let refs: Vec<&BinaryChunked> = ss.iter().map(|s| s.binary().unwrap()).collect_vec();
            merge_ca(&refs, merge_indicator).into_series()
        },
        #[cfg(feature = "dtype-extension")]
        Extension(typ, _) => {
            let storages: Vec<Series> = ss
                .iter()
                .map(|s| s.ext().unwrap().storage().clone())
                .collect();
            merge_series(&storages, merge_indicator)?.into_extension(typ.clone())
        },
        #[cfg(feature = "dtype-struct")]
        Struct(_) => {
            let struct_arrays: Vec<&StructChunked> =
                ss.iter().map(|s| s.struct_().unwrap()).collect_vec();

            let mut validity = None;
            if struct_arrays.iter().any(|s| s.has_nulls()) {
                use arrow::bitmap::Bitmap;

                let validities: Vec<BooleanChunked> = struct_arrays
                    .iter()
                    .map(|s| {
                        BooleanChunked::from_bitmap(
                            PlSmallStr::EMPTY,
                            s.rechunk_validity()
                                .unwrap_or_else(|| Bitmap::new_with_value(true, s.len())),
                        )
                    })
                    .collect();
                let validity_refs: Vec<&BooleanChunked> = validities.iter().collect();
                let mut merged_validity = merge_ca(&validity_refs, merge_indicator);
                merged_validity.rechunk_mut();

                validity = Some(merged_validity.downcast_as_array().values().clone());
            }

            let fields_list: Vec<Vec<Series>> =
                struct_arrays.iter().map(|s| s.fields_as_series()).collect();
            let n_fields = fields_list[0].len();
            let new_fields = (0..n_fields)
                .map(|field_idx| {
                    let field_series: Vec<Series> = fields_list
                        .iter()
                        .map(|fields| fields[field_idx].clone())
                        .collect();
                    let name = fields_list[0][field_idx].name().clone();
                    merge_series(&field_series, merge_indicator)
                        .map(|merged| merged.with_name(name))
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            StructChunked::from_series(PlSmallStr::EMPTY, new_fields[0].len(), new_fields.iter())
                .unwrap()
                .with_outer_validity(validity)
                .into_series()
        },
        #[cfg(feature = "dtype-array")]
        Array(_, _) => {
            // @Optimize. This is horrendous
            let encoded = ss
                .iter()
                .map(|s| s.row_encode_unordered())
                .collect::<PolarsResult<Vec<_>>>()?;
            let fields = std::slice::from_ref(encoded[0].ref_field());
            let refs: Vec<&_> = encoded.iter().collect();
            merge_ca(&refs, merge_indicator)
                .row_decode_unordered(fields)?
                .fields_as_series()
                .pop()
                .unwrap()
        },
        List(_) => {
            // @Optimize. This is horrendous
            let encoded = ss
                .iter()
                .map(|s| s.row_encode_unordered())
                .collect::<PolarsResult<Vec<_>>>()?;
            let fields = std::slice::from_ref(encoded[0].ref_field());
            let refs: Vec<&_> = encoded.iter().collect();
            merge_ca(&refs, merge_indicator)
                .row_decode_unordered(fields)?
                .fields_as_series()
                .pop()
                .unwrap()
        },
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let refs: Vec<&ChunkedArray<$T>> = ss
                    .iter()
                    .map(|s| {
                        let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                        ca
                    })
                    .collect_vec();
                merge_ca(&refs, merge_indicator).into_series()
            })
        },
        dt => polars_bail!(op = "merge_sorted", dt),
    };
    Ok(out)
}

fn merge_ca<'a, T>(cas: &[&'a ChunkedArray<T>], merge_indicator: &[u8]) -> ChunkedArray<T>
where
    T: PolarsDataType + 'static,
    &'a ChunkedArray<T>: IntoIterator,
    T::Array: ArrayFromIterDtype<<&'a ChunkedArray<T> as IntoIterator>::Item>,
{
    let dtype = cas[0].dtype().clone();

    let total_len = cas.iter().map(|ca| ca.len()).sum();
    let mut cas = cas.iter().map(|ca| ca.into_iter()).collect_vec();

    let iter = merge_indicator.iter().map(|indicator| unsafe {
        cas.get_unchecked_mut(*indicator as usize)
            .next()
            .unwrap_unchecked()
    });

    // SAFETY: length is correct
    unsafe {
        iter.trust_my_length(total_len)
            .collect_ca_trusted_with_dtype(PlSmallStr::EMPTY, dtype)
    }
}

fn series_to_merge_indicator(lhs: &Series, rhs: &Series) -> PolarsResult<Vec<u8>> {
    #[cfg(feature = "dtype-categorical")]
    if lhs.dtype().is_categorical() || lhs.dtype().is_enum() {
        let cat_phys = lhs.dtype().cat_physical().unwrap();
        with_match_categorical_physical_type!(cat_phys, |$C| {
            let lhs = lhs.cat::<$C>().unwrap();
            let rhs = rhs.cat::<$C>().unwrap();
            return Ok(get_merge_indicator(lhs.iter_str(), rhs.iter_str()));
        })
    }

    if lhs.dtype().is_nested() {
        return Ok(get_merge_indicator(
            lhs.row_encode_ordered(false, false)?.into_iter(),
            rhs.row_encode_ordered(false, false)?.into_iter(),
        ));
    }

    let lhs_s = lhs.to_physical_repr().into_owned();
    let rhs_s = rhs.to_physical_repr().into_owned();

    let out = match lhs_s.dtype() {
        DataType::Null => vec![0u8; lhs.len() + rhs.len()],
        DataType::Boolean => {
            let lhs = lhs_s.bool().unwrap();
            let rhs = rhs_s.bool().unwrap();
            get_merge_indicator(lhs.into_iter(), rhs.into_iter())
        },
        DataType::Binary => {
            let lhs = lhs_s.binary().unwrap();
            let rhs = rhs_s.binary().unwrap();
            get_merge_indicator(lhs.into_iter(), rhs.into_iter())
        },
        DataType::String => {
            let lhs = lhs.str().unwrap().as_binary();
            let rhs = rhs.str().unwrap().as_binary();
            get_merge_indicator(lhs.into_iter(), rhs.into_iter())
        },
        DataType::BinaryOffset => {
            let lhs = lhs_s.binary_offset().unwrap();
            let rhs = rhs_s.binary_offset().unwrap();
            get_merge_indicator(lhs.into_iter(), rhs.into_iter())
        },
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(lhs_s.dtype(), |$T| {
                    let lhs: &ChunkedArray<$T> = lhs_s.as_ref().as_ref().as_ref();
                    let rhs: &ChunkedArray<$T> = rhs_s.as_ref().as_ref().as_ref();

                    get_merge_indicator(lhs.into_iter(), rhs.into_iter())

            })
        },
        dt => polars_bail!(op = "merge_sorted", dt),
    };
    Ok(out)
}

// Produces an index sequence for merge_ca: 0 = take from a, 1 = take from b.
fn get_merge_indicator<T>(
    mut a_iter: impl ExactSizeIterator<Item = T>,
    mut b_iter: impl ExactSizeIterator<Item = T>,
) -> Vec<u8>
where
    T: PartialOrd + Default + Copy,
{
    const A_INDICATOR: u8 = 0;
    const B_INDICATOR: u8 = 1;

    let a_len = a_iter.size_hint().0;
    let b_len = b_iter.size_hint().0;
    if a_len == 0 {
        return vec![B_INDICATOR; b_len];
    };
    if b_len == 0 {
        return vec![A_INDICATOR; a_len];
    }

    let mut current_a = T::default();
    let cap = a_len + b_len;
    let mut out = Vec::with_capacity(cap);

    let mut current_b = b_iter.next().unwrap();

    for a in &mut a_iter {
        current_a = a;
        if a <= current_b {
            out.push(A_INDICATOR);
            continue;
        }
        out.push(B_INDICATOR);

        loop {
            if let Some(b) = b_iter.next() {
                current_b = b;
                if b >= a {
                    out.push(A_INDICATOR);
                    break;
                }
                out.push(B_INDICATOR);
                continue;
            }
            // b is depleted fill with a indicator
            let remaining = cap - out.len();
            out.extend(std::iter::repeat_n(A_INDICATOR, remaining));
            return out;
        }
    }
    if current_a < current_b {
        out.push(B_INDICATOR);
    }
    // check if current value already is added
    if *out.last().unwrap() == A_INDICATOR {
        out.push(B_INDICATOR);
    }
    // take remaining
    out.extend(b_iter.map(|_| B_INDICATOR));
    assert_eq!(out.len(), b_len + a_len);

    out
}

#[test]
fn test_merge_sorted() {
    fn get_merge_indicator_sliced<T: PartialOrd + Default + Copy>(a: &[T], b: &[T]) -> Vec<u8> {
        get_merge_indicator(a.iter().copied(), b.iter().copied())
    }

    let a = [1, 2, 4, 6, 9];
    let b = [2, 3, 4, 5, 10];

    let out = get_merge_indicator_sliced(&a, &b);
    let expected = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1];
    //               1  2  2  3  4  4  5  6  9  10
    assert_eq!(out, expected);

    // swap
    // it is not the inverse because left is preferred when both are equal
    let out = get_merge_indicator_sliced(&b, &a);
    let expected = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0];
    assert_eq!(out, expected);

    let a = [5, 6, 7, 10];
    let b = [1, 2, 5];
    let out = get_merge_indicator_sliced(&a, &b);
    let expected = [1, 1, 0, 1, 0, 0, 0];
    assert_eq!(out, expected);

    // swap
    // it is not the inverse because left is preferred when both are equal
    let out = get_merge_indicator_sliced(&b, &a);
    let expected = [0, 0, 0, 1, 1, 1, 1];
    assert_eq!(out, expected);
}
