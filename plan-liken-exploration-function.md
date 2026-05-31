# Plan: `Dedupe.explore()` — exploratory duplicate-rate profiling

## Context

Liken can deduplicate, but gives users no way to *assess* whether a dataset needs fuzzy
deduplication at all, or at what threshold. Users currently have to guess thresholds. This
adds an exploratory method — inspired by pandas `DataFrame.describe()` — that reports the
**duplicate rate** per column for exact matching and for fuzzy matching across a sweep of
thresholds, so users can see how aggressively a column collapses as the threshold loosens.

`lk.dedupe(df).explore(["address", "email"])` returns a small describe-like frame:

```
        address  email
exact      0.20   0.10
0.5        0.80   0.60
0.75       0.40   0.30
0.9        0.20   0.12
0.95       0.13   0.10
0.99       0.10   0.10
```

### Decisions (confirmed with the author)
- **Backends:** pandas, polars, **and modin** (modin is pandas-compatible; cheap to include). Any other backend raises a clear `ValueError`.
- **Override semantics:** when `columns` is a dict `{col: deduper}`, the chosen deduper is **swept across `thresholds`** (not used at its own single threshold), keeping a clean rectangular result.
- **Duplicate rate definition:** `rate = (n − n_groups) / n` — the fraction of rows that are redundant duplicates (i.e. would be removed by `drop_duplicates`). For exact this equals pandas `Series.duplicated().mean()`.

## Public API

New method on `Dedupe` ([src/liken/liken.py](src/liken/liken.py)):

```python
def explore(
    self,
    columns: list[str] | dict[str, BaseDeduper],
    *,
    thresholds: list[float] | None = None,   # default [0.5, 0.75, 0.9, 0.95, 0.99]
    frac: float = 1.0,
) -> UserDataFrame:
```

- `columns` — list of column labels (each analysed with the default `lk.fuzzy()`), **or** a dict mapping a column to a single-column **similarity (threshold) deduper** (e.g. `lk.tfidf(ngram=(1,2))`) to sweep for that column. Required.
- `thresholds` — floats in `(0, 1)`; the fuzzy/similarity thresholds to sweep. Defaults to a module constant `DEFAULT_EXPLORE_THRESHOLDS = [0.5, 0.75, 0.9, 0.95, 0.99]`.
- `frac` — float in `(0, 1]`; randomly sample this fraction of rows before analysis (deduplication is ~O(n²), so sampling makes exploration cheap on large data). Default `1.0`.

**Return:** a DataFrame **in the same backend** as the input, rows = metrics (`"exact"` + one per threshold), columns = the analysed columns. pandas/modin set the metric labels as the **index** (true `describe()` feel); polars has no index, so it gets a leading `"metric"` string column (mirrors polars' own `describe()`).

## Implementation

### New module: `src/liken/explore.py`
Holds the heavy lifting so `liken.py` stays thin (mirrors the modular layout: `datasets.py`, `custom.py`, `preprocessors.py`).

- `DEFAULT_EXPLORE_THRESHOLDS: Final = [0.5, 0.75, 0.9, 0.95, 0.99]`
- `run_explore(df, columns, thresholds, frac) -> UserDataFrame`:
  1. `backend = get_backend(df)` ([core/dispatcher.py](src/liken/core/dispatcher.py)); gate `backend.name in {"pandas","polars","modin"}` else raise `ValueError(INVALID_EXPLORE_BACKEND.format(backend.name))`.
  2. Sample (no mutation — returns a new frame): pandas/modin `df.sample(frac=frac)`, polars `df.sample(fraction=frac)`. Skip when `frac == 1.0`.
  3. `wdf = wrap(sampled, id=None)` ([core/dispatcher.py](src/liken/core/dispatcher.py)). Wrapping adds `canonical_id` to a **copy** (pandas `df.assign`, polars immutable) and we only ever read — the user's df is untouched.
  4. Resolve per-column base deduper: list → `fuzzy()` for each; dict → its value (validate it's a `ThresholdDeduper`).
  5. For each column: compute the **exact** rate once (`exact()`), then for each threshold `t` a swept rate.
  6. Assemble rows `["exact"|str(t), rate_col1, rate_col2, ...]` and build the result with `backend.create_df(data=rows, schema=["metric", *col_names])`; for pandas/modin `result = result.set_index("metric")`.

- `_duplicate_rate(deduper, wdf, column) -> float` (the reusable core):
  ```python
  uf, n = deduper.set_frame(wdf).build_union_find(column, [])   # core/deduper.py:103
  if n == 0:
      return 0.0
  n_groups = len({uf[i] for i in range(n)})
  return (n - n_groups) / n
  ```
- `_at_threshold(base, t)`: `d = copy.copy(base); d._threshold = t; return d` — all built-in similarity dedupers (`Fuzzy`, `TfIdf`, `LSH`, `Jaccard`, `Cosine`) read `self._threshold` at run time, so this is sufficient and avoids fragile re-construction. (We set `_threshold` directly, bypassing `ThresholdDeduper.__init__`'s range check, so our validator enforces the `(0,1)` range — see below.)

Reused as-is (do **not** reimplement): `BaseDeduper.set_frame` / `build_union_find` ([core/deduper.py](src/liken/core/deduper.py)), `get_backend` / `wrap` ([core/dispatcher.py](src/liken/core/dispatcher.py)), `exact()` ([dedupers/exact.py](src/liken/dedupers/exact.py)), `fuzzy()` ([dedupers/fuzzy.py](src/liken/dedupers/fuzzy.py)), `backend.create_df` ([core/backend.py](src/liken/core/backend.py)).

### `src/liken/liken.py`
Add the thin `explore` method on `Dedupe`: validate args, then `return run_explore(self._df, columns, thresholds or DEFAULT_EXPLORE_THRESHOLDS, frac)`. Google-style docstring matching `canonicalize` (Args/Returns/Raises). No `__all__`/`__init__.py` change needed — mkdocstrings auto-documents public `Dedupe` methods.

### `src/liken/validators.py` + `src/liken/constants.py`
Follow the existing `validate_*` + `INVALID_*` idiom (raise `ValueError`/`TypeError` with a constant message):
- `validate_frac_arg(frac)` → float in `(0, 1]` else `ValueError(INVALID_FRAC.format(frac))`.
- `validate_thresholds_arg(thresholds)` → non-empty list, every value a float in `(0, 1)` else `ValueError(INVALID_THRESHOLDS.format(thresholds))`.
- `validate_explore_columns_arg(columns, df_columns)` → non-empty `list[str]` or `dict[str, BaseDeduper]`; dict values must be `ThresholdDeduper` (similarity) else `ValueError(INVALID_EXPLORE_DEDUPER...)`; every referenced column must exist in the df else `ValueError`. (Reuse `validate_columns_arg` style.)
- New constants: `INVALID_EXPLORE_BACKEND`, `INVALID_FRAC`, `INVALID_THRESHOLDS`, `INVALID_EXPLORE_COLUMNS`, `INVALID_EXPLORE_DEDUPER`.

### Conventions to honour
`from __future__ import annotations`; keyword-only args after `*`; `|` unions; ruff line-length 120, single-line imports; no mutable default (resolve `thresholds=None` inside).

## Edge cases
- `frac` samples to 0 rows → rates `0.0` (guard `n == 0`).
- Compound/predicate dedupers in the dict → rejected by `validate_explore_columns_arg` (explore is single-column similarity only; `jaccard`/`cosine` need column tuples — out of scope for v1).
- Nulls: exact/fuzzy use `with_na_placeholder=True`, so nulls coalesce to a placeholder and group together (consistent with `drop_duplicates`).
- `thresholds` containing `1.0` → rejected (deduper semantics require `< 1`).
- Custom dedupers (`_Custom`) are `ThresholdDeduper`s but may ignore `_threshold`; sweeping then yields a flat row — acceptable, note in docstring.

## Tests — `tests/unit/test_explore.py`
Uses the existing `--backend` harness (`dataframe` fixture = `fake_10(backend)`, `helpers.backend`):
- **Supported backends** (`pandas`/`polars`/`modin`): on `fake_10`, `explore(["address","email"])` (default, `frac=1.0`) returns shape `(1+len(thresholds), 2)`; assert deterministic exact rates — **email = 0.1** (`b@example.com` twice) and **address = 0.2** (`"123ab, OL5 9PL, UK"` twice + two nulls→placeholder). Assert every rate in `[0,1]` and non-increasing as threshold rises.
- **Dict override sweep**: `explore({"address": lk.tfidf()})` → same row layout, `address` column present.
- **`frac`**: `frac=0.5` → correct shape, rates in `[0,1]`.
- **Unsupported backends** (`pyspark`/`dask`/`ray`): `with pytest.raises(ValueError): dupe.explore([...])` (gate happens before any heavy work). Branch on `helpers.backend`.
- **Validation**: bad `frac` (0, 1.5), bad `thresholds` ([], [1.0]), bad `columns` (empty list, predicate deduper in dict, missing column) → raises.

## Docs (light)
- API reference auto-updates via mkdocstrings (no action).
- Add a feature bullet to [docs/index.md](docs/index.md) ("Exploratory duplicate-rate profiling with `.explore`").
- Optional: a short "Exploring your data" subsection in [docs/tutorials/first-steps.md](docs/tutorials/first-steps.md) before applying dedupers.
- Out of scope (note for later): updating the `liken-skills` bundle to mention `.explore`.

## Verification
1. **Unit tests** across the three supported backends:
   `uv run pytest tests/unit/test_explore.py --backend=pandas` (then `--backend=polars`, `--backend=modin`); plus one unsupported, e.g. `--backend=dask`, to confirm the `ValueError` gate.
2. **Manual smoke** with the `.venv` python on `fake_10`:
   ```python
   import liken as lk
   print(lk.dedupe(lk.datasets.fake_10()).explore(["address", "email"]))            # pandas, describe-like, exact=0.2/0.1
   print(lk.dedupe(lk.datasets.fake_10("polars")).explore(["address"], frac=0.5))   # polars: leading "metric" column
   print(lk.dedupe(lk.datasets.fake_10()).explore({"address": lk.tfidf(ngram=(1,2))}))
   ```
   Confirm: correct shape, exact rates as above, rates within `[0,1]`, non-increasing with threshold, and that the original `df` is unchanged (no `canonical_id` leaked).
3. **Lint/type**: `uv run ruff check` and `uv run mypy src/liken/explore.py src/liken/liken.py`.
