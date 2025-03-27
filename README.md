A Python library for grouping duplicate data efficiently.

# Introduction
**DuperGrouper** can be used for various deduplication use cases. It's intended purpose is to implement a uniform API that allows for both exact *and* near deduplication — whilst collecting duplicate instances into sets — i.e. "groups".

Deduplicating data is a hard task — validating approaches takes time, can require a lot of scripting, testing, and iterating through approaches that may, or may not, be valid to your dataset.

**DuperGrouper** abstracts away the task *actually* deduplicating, so that you can focus on the most important thing: implementing an appropriate "strategy" to achieve your stated end goal ...

...In fact a "strategy" is key to **DupeGrouper's** API. **DupeGrouper** has:

- Ready-to-use deduplication strategies
- Pandas and Polars support
- A flexible API

Head to the API Reference [here](api/index.html).

## Installation


```shell
pip install dupegrouper
```

## Example

```python
import dupegrouper

dg = dupegrouper.DupeGrouper(df) # input dataframe

dg.dedupe("address")

dg.df # retrieve dataframe
```

# Usage Guides

## Adding Strategies

The base method `dedupe()` implements an **exact** deduplication by default. You can add `dupegrouper` strategies with the `add_strategy()` method:

```python
# Deduplicate the address column

dg = dupegrouper.DupeGrouper(df)

dg.add_strategy(dupegrouper.Exact())
dg.add_strategy(dupegrouper.Fuzzy(tolerance=0.3))

dg.dedupe("address")
```

The `add_strategy` method is flexible: add strategies in bulk, if you wish:

```python
# Also deduplicates the address column

dg = dupegrouper.DupeGrouper(df)

dg.add_strategy({
    "address": [
        dupegrouper.Exact(),
        dupegrouper.Fuzzy(tolerance=0.3),
    ]
})

dg.dedupe() # No Argument!
```

## Custom Strategies

`DupeGrouper` can accept custom functions too. Your callable is added as strategy and passed to the `Custom` class, inheriting all of the base functionality of `DeduplicationStrategy`. E.g.

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

Above, `my_func` deserves a custom implementation: it deduplicates rows only if said rows contain a the partial string `match_str`. You can your custom function as a strategy:

```python
dg = dupegrouper.DupeGrouper(df)

dg.add_strategy((my_func, {"match_str": "london"}))

print(dg.strategies) # returns ("my_func",)

dg.dedupe("address")
```

> [!IMPORTANT]  
> Your custom function's signatured must be as above: 
> 
> ```python
> (df: pd.DataFrame, attr: str, /, match_str: str) -> dict[str, str]
> ```

## Creating a Comprehensive Strategy

You can collect all of the above's ideas for a comprehensive strategy to deduplicate your data:

```python
import dupegrouper
import pandas

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

## Extending API for Custom Implementations
It's recommended that for simple custom implementations you use the approach discussed for custom functions. (see [*Custom Strategies*](#custom-strategies))

However, you can also inherit the TODO, and make direct use of TODO in a custom class implementation, and thus expose a custom `dedupe()` method, ready for use with an instance of `DupeGrouper`

# About

## License
This project is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0). See the [LICENSE](LICENSE) file for more details.