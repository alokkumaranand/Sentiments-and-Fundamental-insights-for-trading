"""Statistical helpers. Pure functions over numpy arrays; no dataframes, no config, no I/O.

The two decisions that matter are in `inference.py`: sessions, not rows, are the resampling unit,
and every p-value that reaches a report is BH-adjusted for its batch.
"""
