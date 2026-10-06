# Development checks

Run from the repository root after setup:

```bash
uv pip install --python .venv/bin/python -e '.[dev,report]'
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m unittest discover -s tests -t . -v
```

Use `./.tools/uv` if setup installed uv locally. Tests create small models and
datasets in temporary directories. No previous Git commits, measurement archives
or model downloads are needed. Optional Torch and COCO checks run when those
dependencies are installed.

`fixtures/pot/golden_40.json` records expected graph and output hashes for a
synthetic model. The quantization suite exercises 480 scale/policy combinations.
`fixtures/public_api_signatures.json` defines the public API contract.
