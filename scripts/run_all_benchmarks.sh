#!/usr/bin/env bash
# Full sweep. Writes results_*.json (gitignored -- they embed a GPU fingerprint).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-lm1b}"
echo "=== analytic cost model ($CONFIG) ==="
python -m bench.roofline --config "$CONFIG"

echo; echo "=== correctness ==="
pytest -q

echo; echo "=== op sweep, forward ==="
python -m bench.bench_ops --config "$CONFIG" --out "results_ops_${CONFIG}_fwd.json"

echo; echo "=== op sweep, forward+backward ==="
python -m bench.bench_ops --config "$CONFIG" --backward \
    --out "results_ops_${CONFIG}_fwdbwd.json"

echo; echo "Done. Promote runs worth keeping into docs/results/ by hand."
