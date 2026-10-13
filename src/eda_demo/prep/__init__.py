"""Preparation layer: raw rows in, canonical tables and a ticker-session panel out.

Nothing here is dropped for being low quality — rows are flagged, counted and reported
(SPEC.md 4.2), because a silently dropped row changes every downstream number without
changing any reported count.
"""
