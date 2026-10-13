# Data quality report

- Dataset slice: **dev**
- Tweets: `C:\My library\MyProjects\Little Ones\EDAProject\splits\tweets_validation.csv`
- Prices: `C:\My library\MyProjects\Little Ones\EDAProject\splits\prices_validation.csv`
- `data_version`: `9869d568e7ff`

## Inputs and hashes

| File | SHA-256 |
|---|---|
| tweets (container) | `6a4a9642d0d16782fe8c6d06b32f4662b525cbddc58eea176591e6056519bce5` |
| prices (container) | `e1d6cd294aa33a5a1718be00c272c965478d688633eca849bb75affab0e84032` |

## Tweets

- 8,079 rows x 4 cols; 0 null cells
- Window: 2021-09-30T00:06:02+00:00 -> 2021-11-02T07:29:00+00:00
- 25 tickers; 6,779 distinct texts
- Duplicate rows: 0; duplicate (ticker, text): 10
- **Cross-ticker fan-out: 541 texts appear under >1 ticker, covering 1,831 rows (22.7%)**
- Promo-shaped (URL or >=4 cashtags): 4,389 (54.3%)
- HTML entities present in 746 rows
- Replies (start with @): 916; cashtags: 6,189; hashtags: 1,128
- On a non-trading date: 1,682 (20.8%) -> rolls to the next session
- At/after the close hour: 3,018 (37.4%) -> rolls to the next session
- Text length (chars): min 10, median 142.0, max 445

### Tweets per ticker, own-text share, tier

| Ticker | Tweets | Own-text share | Tier |
|---|---:|---:|:--:|
| TSLA | 4,334 | 0.95 | A |
| TSM | 1,070 | 0.99 | A |
| META | 448 | 0.73 | B |
| AAPL | 345 | 0.41 | B |
| AMZN | 287 | 0.00 | C |
| MSFT | 287 | 0.00 | C |
| PG | 287 | 0.00 | C |
| NIO | 281 | 0.78 | B |
| AMD | 192 | 0.66 | B |
| NFLX | 111 | 0.40 | B |
| GOOG | 96 | 0.54 | B |
| PYPL | 77 | 0.70 | B |
| BA | 48 | 0.44 | B |
| DIS | 41 | 0.54 | B |
| XPEV | 37 | 0.65 | B |
| KO | 22 | 0.18 | B |
| ENPH | 22 | 0.41 | B |
| INTC | 21 | 0.33 | B |
| COST | 19 | 0.63 | B |
| CRM | 18 | 0.33 | B |
| ZS | 15 | 0.73 | B |
| VZ | 13 | 0.23 | B |
| BX | 4 | 0.50 | B |
| F | 2 | 0.00 | C |
| NOC | 2 | 0.00 | C |

## Prices

- 630 rows x 8 cols; 0 null cells
- Window: 2021-09-30T00:00:00+00:00 -> 2021-11-04T00:00:00+00:00 (26 sessions, 25 tickers)
- Duplicate (ticker, date): 0; weekend rows: 0
- Rows per ticker: min 25, max 26, values [25, 26]
- Ragged sessions (not all tickers present): 1 ['2021-11-04']
- `Adj Close` equals `Close` on 353 rows; max difference 7.5978
- **Split continuity: passed** — largest |1-day return| 0.2465 (ENPH), limit 0.5

## Reconciliation

- Tickers shared by both files: 25
- Only in tweets: none
- Only in prices: none

## Tier summary

- Thresholds: {'specific_own_text_share': 0.8, 'mixed_own_text_share': 0.1}
- Tier A (specific): 2 | Tier B (mixed): 18 | Tier C (generic): 5

## Warnings

- 1 price session(s) are incomplete (not all tickers present): ['2021-11-04']. Drop them before computing any cross-sectional statistic.
- 5 ticker(s) are coverage tier C (100% cross-posted text): ['AMZN', 'F', 'MSFT', 'NOC', 'PG']. They are excluded from sentiment and attention inference.
- only 26 trading session(s) in this dataset. Trailing-window features (20-session baselines) and any discovery/holdout split are not meaningful. Use --dataset full for real analysis.
- 746 tweet(s) contain HTML entities and need html.unescape in cleaning.
- 54.3% of tweets are promo-shaped (URL or >=4 cashtags).
- 20.8% of tweets fall on a non-trading date and roll forward to the next session.

## Blocking errors

- none
