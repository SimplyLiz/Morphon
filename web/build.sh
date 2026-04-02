#!/bin/bash
# Build MORPHON WASM package and prepare web demo
set -e

cd "$(dirname "$0")/.."

echo "Building WASM..."
wasm-pack build --target web --features wasm --no-default-features

echo "Moving pkg to web/..."
rm -rf web/pkg
mv pkg web/pkg

echo "Done. Serve with:"
echo "  cd web && python3 serve.py"
