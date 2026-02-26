# Archive Notes

`BabyJouncer.ipynb` is retained as a legacy exploration notebook from December 2017.

## What Changed

- Removed embedded Google Maps API key material.
- Notebook now reads `GOOGLE_MAPS_API_KEY` from environment variables.

## Why This File Exists

- Preserve original exploratory work without pretending it is production-ready.
- Document known limitations and reduce accidental secret leakage.

## Known Legacy Limitations

- Notebook-first workflow, not modular code.
- No dependency lockfile from the original era.
- Multiple exploratory/incomplete cells.
- Activity labels in source data are noisy and should not be treated as clean ground truth.
