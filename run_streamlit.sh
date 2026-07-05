#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/work/ariadne-env/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Expected Python environment at $PYTHON"
  echo "Create it with: ./setup_env.sh"
  exit 1
fi

cd "$ROOT"
exec "$PYTHON" -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
