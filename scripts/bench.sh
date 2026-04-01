#!/usr/bin/env zsh
#
# Morphon Benchmark Suite Runner
#
# Usage:
#   ./scripts/bench.sh                          # quick profile, all benchmarks
#   ./scripts/bench.sh --standard               # standard profile
#   ./scripts/bench.sh --extended               # extended profile
#   ./scripts/bench.sh --quick --skip-mnist     # skip MNIST (no data dir)
#   ./scripts/bench.sh --skip-criterion         # skip Criterion micro-benchmarks
#   ./scripts/bench.sh --report-only            # regenerate report from existing results
#
set -euo pipefail

# --- Parse arguments ---
PROFILE="quick"
SKIP_MNIST=false
SKIP_CRITERION=false
REPORT_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --quick)          PROFILE="quick" ;;
        --standard)       PROFILE="standard" ;;
        --extended)       PROFILE="extended" ;;
        --skip-mnist)     SKIP_MNIST=true ;;
        --skip-criterion) SKIP_CRITERION=true ;;
        --report-only)    REPORT_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--quick|--standard|--extended] [--skip-mnist] [--skip-criterion] [--report-only]"
            exit 0 ;;
        *)
            echo "Unknown argument: $arg"
            exit 1 ;;
    esac
done

# --- Get version from Cargo.toml ---
VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
RESULTS_DIR="docs/benchmark_results/v${VERSION}"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "  Morphon Benchmark Suite v${VERSION}"
echo "  Profile: ${PROFILE}"
echo "  Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "========================================"
echo ""

if $REPORT_ONLY; then
    echo ">> Regenerating report from existing results..."
    cargo run --bin bench_report --release -- --profile "$PROFILE"
    echo ""
    echo "Done. Report at: ${RESULTS_DIR}/REPORT.md"
    exit 0
fi

FAILURES=0

# --- Step 1: Criterion Micro-Benchmarks ---
if ! $SKIP_CRITERION; then
    echo ">> [1/4] Running Criterion micro-benchmarks..."
    echo ""
    if cargo bench 2>&1 | tee /tmp/morphon_bench_output.txt; then
        echo ""
        echo ">> Extracting Criterion results..."
        extract_criterion_results
    else
        echo "WARNING: Criterion benchmarks failed"
        FAILURES=$((FAILURES + 1))
    fi
    echo ""
else
    echo ">> [1/4] Skipping Criterion (--skip-criterion)"
    echo ""
fi

# --- Step 2: CartPole ---
echo ">> [2/4] Running CartPole benchmark (${PROFILE})..."
echo ""
if cargo run --example cartpole --release -- --${PROFILE} 2>&1; then
    echo ""
else
    echo "WARNING: CartPole benchmark failed"
    FAILURES=$((FAILURES + 1))
fi

# --- Step 3: Anomaly Detection ---
echo ">> [3/4] Running Anomaly Detection benchmark (${PROFILE})..."
echo ""
if cargo run --example anomaly --release -- --${PROFILE} 2>&1; then
    echo ""
else
    echo "WARNING: Anomaly benchmark failed"
    FAILURES=$((FAILURES + 1))
fi

# --- Step 4: MNIST ---
if ! $SKIP_MNIST; then
    if [ -d "data" ]; then
        echo ">> [4/4] Running MNIST benchmark (${PROFILE})..."
        echo ""
        if cargo run --example mnist --release -- --${PROFILE} 2>&1; then
            echo ""
        else
            echo "WARNING: MNIST benchmark failed"
            FAILURES=$((FAILURES + 1))
        fi
    else
        echo ">> [4/4] Skipping MNIST (no ./data/ directory found)"
        echo "   Download MNIST data to ./data/ to enable this benchmark."
        echo ""
    fi
else
    echo ">> [4/4] Skipping MNIST (--skip-mnist)"
    echo ""
fi

# --- Step 5: Generate Report ---
echo ">> Generating report..."
cargo run --bin bench_report --release -- --profile "$PROFILE"

echo ""
echo "========================================"
echo "  Benchmark Suite Complete"
if [ $FAILURES -gt 0 ]; then
    echo "  Warnings: ${FAILURES} benchmark(s) failed"
fi
echo "  Results:  ${RESULTS_DIR}/"
echo "  Report:   ${RESULTS_DIR}/REPORT.md"
echo "========================================"

# --- Helper: Extract Criterion results to JSON ---
extract_criterion_results() {
    local criterion_dir="target/criterion"
    local output="${RESULTS_DIR}/criterion_latest.json"

    if [ ! -d "$criterion_dir" ]; then
        echo "  No Criterion cache found at $criterion_dir"
        return
    fi

    # Build JSON object from estimates.json files
    echo "{" > "$output"
    local first=true

    for bench_dir in "$criterion_dir"/*/; do
        local bench_name=$(basename "$bench_dir")
        [ "$bench_name" = "report" ] && continue

        local estimates="$bench_dir/new/estimates.json"
        if [ -f "$estimates" ]; then
            # Flat benchmark
            $first || echo "," >> "$output"
            first=false
            write_bench_entry "$bench_name" "$estimates" >> "$output"
        else
            # Grouped benchmark — scan subdirectories
            for param_dir in "$bench_dir"/*/; do
                local param=$(basename "$param_dir")
                [ "$param" = "report" ] && continue
                local param_estimates="$param_dir/new/estimates.json"
                if [ -f "$param_estimates" ]; then
                    $first || echo "," >> "$output"
                    first=false
                    write_bench_entry "${bench_name}/${param}" "$param_estimates" >> "$output"
                fi
            done
        fi
    done

    echo "" >> "$output"
    echo "}" >> "$output"
    echo "  Criterion results saved to $output"
}

write_bench_entry() {
    local name="$1"
    local file="$2"

    # Extract values using python3 (available on macOS)
    python3 -c "
import json, sys
with open('$file') as f:
    d = json.load(f)
entry = {
    'mean': d['mean']['point_estimate'],
    'median': d['median']['point_estimate'],
    'std_dev': d['std_dev']['point_estimate']
}
# Output without trailing newline
sys.stdout.write('  \"$name\": ' + json.dumps(entry))
"
}
