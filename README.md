A Python library for grouping duplicate data efficiently.

<p align="center">
<a href="https://pypi.python.org/pypi/dupegrouper"><img height="20" alt="PyPI Version" src="https://img.shields.io/pypi/v/dupegrouper"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dupegrouper">
</p>

# Introduction

**dupegrouper** can be used for various deduplication use cases. It's intended purpose is to implement a uniform API that allows for both exact *and* near deduplication — whilst collecting duplicate instances into sets — i.e. "groups".

Deduplicating data is a hard task — validating approaches takes time, can require a lot of testing, validating, and iterating through approaches that may, or may not, be applicable to your dataset.

**dupegrouper** abstracts away the task of *actually* deduplicating, so that you can focus on the most important thing: implementing an appropriate "strategy" to achieve your stated end goal ...

...In fact a "strategy" is key to **dupegrouper's** API. **dupegrouper** has:

- Ready-to-use deduplication strategies
- Pandas and Polars support
- A flexible API

Checkout the [API Documentation](https://victorautonell-oiry.me/dupegrouper/dupegrouper.html).


## Installation


```shell
pip install dupegrouper
```

## Example

```python
import dupegrouper

dg = dupegrouper.DupeGrouper(df) # input dataframe

dg.add_strategy(dupegrouper.strategies.Exact())

dg.dedupe("address")

dg.df # retrieve dataframe
```

# Usage Guide

## Adding Strategies

**dupegrouper** comes with ready-to-use deduplication strategies:
- `dupegrouper.strategies.Exact`
- `dupegrouper.strategies.Fuzzy`
- `dupegrouper.strategies.TfIdf`

You can then add these in the order you want to apply them:

```python
# Deduplicate the address column

dg = dupegrouper.DupeGrouper(df)

dg.add_strategy(dupegrouper.strategies.Exact())
dg.add_strategy(dupegrouper.strategies.Fuzzy(tolerance=0.3))

dg.dedupe("address")
```

Or, add a map of strategies:

```python
# Also deduplicates the address column

dg = dupegrouper.DupeGrouper(df)

dg.add_strategy({
    "address": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.3),
    ]
})

dg.dedupe() # No Argument!
```

## Custom Strategies

An insance of `dupegrouper.DupeGrouper` can accept custom functions too.

```python
def my_func(df: pd.DataFrame, attr: str, /, match_str: str) -> dict[str, str]:
    """deduplicates df if any given row contains `match_str`"""
    my_map = {}
    for irow, _ in df.iterrows():
        left: str = df.at[irow, attr]
        my_map[left] = left
        for jrow, _ in df.iterrows():
            right: str = df.at[jrow, attr]
            if match_str in left.lower() and match_str in right.lower():
                my_map[left] = right
                break
    return my_map
```

Above, `my_func` deserves a custom implementation: it deduplicates rows only if said rows contain a the partial string `match_str`. You can then proceed to add your custom function as a strategy:

```python
dg = dupegrouper.DupeGrouper(df)

dg.add_strategy((my_func, {"match_str": "london"}))

print(dg.strategies) # returns ("my_func",)

dg.dedupe("address")
```

> [!NOTE]
> Your custom function's signature must be two positional arguments followed by keyword arguments:
> 
> ```python
> (df: DataFrame, attr: str, /, **kwargs) -> dict[str, str]
> ```
>
> Where `attr` is the attribute you wish to deduplicate.

> [!WARNING]
> In the current implementation, any custom callable will also *always dedupe exact matches!*

## Creating a Comprehensive Strategy

You can use the above techniques for a comprehensive strategy to deduplicate your data:

```python
import dupegrouper
import pandas # or polars

df = pd.read_csv("example.csv")

dg = dupegrouper.DupeGrouper(df)

strategies = {
    "address": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.5),
        (my_func, {"match_str": "london"}),
    ],
    "email": [
        dupegrouper.strategies.Exact(),
        dupegrouper.strategies.Fuzzy(tolerance=0.3),
        dupegrouper.strategies.TfIdf(tolerance=0.4, ngram=3, topn=2),
    ],
}

dg.add_strategy(strategies)

dg.dedupe()

df = dg.df
```

## Extending the API for Custom Implementations
It's recommended that for simple custom implementations you use the approach discussed for custom functions. (see [*Custom Strategies*](#custom-strategies)).

However, you can derive directly from the abstract base class `dupegrouper.strategy.DeduplicationStrategy`, and thus make direct use of the efficient, core deduplication methods implemented in this library, as described in it's [API](./dupegrouper/strategy.html#DeduplicationStrategy). This will expose a `dedupe()` method, ready for direct use within an instance of `DupeGrouper`, much the same way that other `dupegrouper.strategies` are passed in as strategies.

# About

## License
This project is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html). See the [LICENSE](LICENSE) file for more details.