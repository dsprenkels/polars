import importlib

import pytest

import polars as pl
from polars.exceptions import AttributeRemovedError


def test_init_nonexistent_attribute() -> None:
    with pytest.raises(
        AttributeError, match="module 'polars' has no attribute 'stroopwafel'"
    ):
        pl.stroopwafel  # type: ignore[attr-defined]


def test_init_exceptions_deprecated() -> None:
    with pytest.raises(
        AttributeRemovedError,
        match=r"accessing `ComputeError` from the top-level `polars` module was deprecated in version 1\.0\.0",
    ):
        pl.ComputeError  # type: ignore[attr-defined]


def test_dtype_groups_deprecated() -> None:
    with pytest.raises(
        AttributeRemovedError,
        match=r"`INTEGER_DTYPES` was deprecated in version 1\.0\.0",
    ):
        pl.INTEGER_DTYPES  # type: ignore[attr-defined]


def test_type_aliases_deprecated() -> None:
    with pytest.deprecated_call(
        match=r"the `polars\.type_aliases` module was deprecated in version 1.0.0."
    ):
        from polars.type_aliases import PolarsDataType

        _ = PolarsDataType


def test_import_all() -> None:
    exec("from polars import *")


def test_version() -> None:
    # This has already gone wrong once (#23940), preventing future problems.
    lhs = pl.__version__.replace("-beta.", "b")
    rhs = importlib.metadata.version("polars")

    assert lhs == rhs, (
        f"`static PYPOLARS_VERSION` ({lhs}) at `crates/polars-python/src/c_api/mod.rs` "
        f"does not match importlib package metadata version ({rhs})"
    )
