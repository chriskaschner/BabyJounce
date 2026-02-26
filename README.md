# BabyJounce

Never ever, ever, ever, ever, ever jounce a baby.

![Fritzy](img/Fritz-6767.jpg "Fritzasaurus")

## Project Status

This repo now has two tracks:

1. Legacy archive notebook: `BabyJouncer.ipynb`
2. Reproducible v2 analysis CLI: `src/babyjounce/`

## 1) Legacy Archive (Notebook)

The original notebook remains for historical context.

- A hardcoded Google Maps key was removed from `BabyJouncer.ipynb`.
- Notebook map rendering now reads `GOOGLE_MAPS_API_KEY` from environment variables.
- If no key is set, gmaps layers will not authenticate.

See `ARCHIVE.md` for archive notes and guardrails.

## 2) Updated Version (v2, Reproducible)

The v2 path is a small, testable Python package (stdlib only).

### Run the analysis

```bash
PYTHONPATH=src python3 -m babyjounce --data-dir data
```

Or use the wrapper script:

```bash
python3 scripts/run_analysis.py --data-dir data
```

### Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### Optional install as CLI

```bash
python3 -m pip install -e .
babyjounce --data-dir data
```
