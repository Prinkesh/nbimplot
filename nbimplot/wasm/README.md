# nbimplot WASM Artifacts

This directory receives generated files from:

```bash
scripts/build_wasm.sh
```

Expected outputs:

- `nbimplot_wasm.js`
- `nbimplot_wasm.wasm`

These artifacts are optional at import time. If they are missing, the widget uses the JS fallback renderer.
