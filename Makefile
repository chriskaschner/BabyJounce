.PHONY: analyze test

analyze:
	PYTHONPATH=src python3 -m babyjounce --data-dir data

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v
