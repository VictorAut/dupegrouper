# Welcome to DupeGrouper

DupeGrouper is a Python library for grouping duplicate data efficiently.

## Introduction
- Custom deduplication strategies
- Pandas and Polars support
- Flexible API

Head to the API Reference [here](api/index.html).

## Installation


```shell
pip install dupegrouper
```

# Usage

```python
import dupegrouper

dg = dupegrouper.DupeGrouper(df) # input dataframe

dg.add_strategy(dupegrouper.strategies.Exact())
dg.add_strategy(dupegrouper.strategies.Fuzzy(tolerance=0.3))

dg.dedupe("address")

dg.df # retrieve dataframe
```