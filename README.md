# Anime Ranker (React + Deno)

Local React UI to rank anime from MAL exports with fast hotkeys and Elo math.

## What it does

1. Loads a MyAnimeList XML export (`.xml` or `.xml.gz`)
2. Lets you rank anime via Elo-style “A vs B” comparisons
3. Fits a Normal distribution to the resulting Elo ratings
4. Converts each anime’s Elo to a discretized `1..10` score
5. Exports results as CSV/JSON (optionally writes to disk via File System Access API)

## Run

```bash
deno task install
deno task dev --host 127.0.0.1 --port 5173
```

## Notes

- `.gz` is decompressed in-browser.
- The `1..10` score is computed by:
  - fitting `(mu, sigma)` from all Elo values,
  - computing each anime’s percentile with `NormalCDF((elo-mu)/sigma)`,
  - mapping percentile → decile bucket `1..10`.
- Optional posters/English titles use the public Jikan API (no MAL OAuth required).
- Optional assumption: dropped titles are placed below non-dropped and comparisons between Completed/Dropped are skipped.
