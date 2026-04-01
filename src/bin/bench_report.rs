//! Benchmark Report Generator — reads JSON results and produces a Markdown report.
//!
//! Usage: cargo run --bin bench_report --release [-- --profile standard]
//!
//! Reads from docs/benchmark_results/v{version}/ and writes REPORT.md there.

use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let profile = parse_profile();
    let results_dir = format!("docs/benchmark_results/v{VERSION}");

    if !Path::new(&results_dir).exists() {
        eprintln!("No results directory found at {results_dir}");
        eprintln!("Run ./scripts/bench.sh first to generate benchmark data.");
        std::process::exit(1);
    }

    let mut report = String::new();

    write_header(&mut report, &profile);
    write_criterion_section(&mut report, &results_dir);
    write_scaling_section(&mut report, &results_dir);
    write_cartpole_section(&mut report, &results_dir);
    write_anomaly_section(&mut report, &results_dir);
    write_mnist_section(&mut report, &results_dir);
    write_comparison_section(&mut report, &results_dir);
    write_footer(&mut report);

    let report_path = format!("{results_dir}/REPORT.md");
    fs::write(&report_path, &report).expect("Failed to write report");
    println!("Report written to {report_path}");

    // Generate HTML dashboard
    let html = generate_html_dashboard(&results_dir, &profile);
    let html_path = format!("{results_dir}/REPORT.html");
    fs::write(&html_path, &html).expect("Failed to write HTML report");
    println!("Dashboard written to {html_path}");
}

fn parse_profile() -> String {
    std::env::args()
        .skip_while(|a| a != "--profile")
        .nth(1)
        .unwrap_or_else(|| "quick".to_string())
}

fn load_json(path: &str) -> Option<Value> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn fmt_time(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.1} us", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000_000.0)
    }
}

fn fmt_delta(current: f64, previous: f64) -> String {
    if previous == 0.0 {
        return "N/A".to_string();
    }
    let pct = (current - previous) / previous * 100.0;
    if pct < -1.0 {
        format!("{pct:+.1}%")
    } else if pct > 1.0 {
        format!("{pct:+.1}%")
    } else {
        "~0%".to_string()
    }
}

// === Header ===

fn write_header(report: &mut String, profile: &str) {
    let now = chrono_lite();
    report.push_str(&format!("# Morphon Benchmark Report\n\n"));
    report.push_str(&format!("| | |\n|---|---|\n"));
    report.push_str(&format!("| **Version** | {VERSION} |\n"));
    report.push_str(&format!("| **Profile** | {profile} |\n"));
    report.push_str(&format!("| **Date** | {now} |\n"));
    report.push_str(&format!("| **Platform** | {} {} |\n\n", std::env::consts::OS, std::env::consts::ARCH));
}

fn chrono_lite() -> String {
    // Simple timestamp without chrono dependency
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // Convert to readable date (good enough without chrono)
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;

    // Days since epoch to Y-M-D (simplified)
    let (year, month, day) = days_to_date(days);
    format!("{year}-{month:02}-{day:02} {hours:02}:{minutes:02} UTC")
}

fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Simplified Gregorian calendar conversion
    let mut y = 1970;
    let mut remaining = days;
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }
    let month_days = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m = 0;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining < md {
            m = i;
            break;
        }
        remaining -= md;
    }
    (y, m as u64 + 1, remaining + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// === Criterion Micro-Benchmarks ===

fn write_criterion_section(report: &mut String, results_dir: &str) {
    let criterion_path = format!("{results_dir}/criterion_latest.json");
    let Some(data) = load_json(&criterion_path) else {
        report.push_str("## Micro-Benchmarks\n\n*No Criterion results found. Run `cargo bench` first.*\n\n");
        return;
    };

    report.push_str("## Micro-Benchmarks\n\n");
    report.push_str("| Benchmark | Median | Mean | Std Dev |\n");
    report.push_str("|-----------|--------|------|---------|\n");

    let Some(benchmarks) = data.as_object() else { return };

    // Sort by name for consistent output
    let mut entries: Vec<_> = benchmarks.iter().collect();
    entries.sort_by_key(|(k, _)| k.to_string());

    for (name, values) in entries {
        let median = values["median"].as_f64().unwrap_or(0.0);
        let mean = values["mean"].as_f64().unwrap_or(0.0);
        let std_dev = values["std_dev"].as_f64().unwrap_or(0.0);

        report.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            name,
            fmt_time(median),
            fmt_time(mean),
            fmt_time(std_dev),
        ));
    }
    report.push('\n');
}

// === Scaling Analysis ===

fn write_scaling_section(report: &mut String, results_dir: &str) {
    let criterion_path = format!("{results_dir}/criterion_latest.json");
    let Some(data) = load_json(&criterion_path) else { return };
    let Some(benchmarks) = data.as_object() else { return };

    // Find scaling benchmarks (system_step_scaling/* and resonance_propagate/*)
    let scaling_groups: Vec<(&str, &str)> = vec![
        ("system_step_scaling", "System Step"),
        ("resonance_propagate", "Resonance Propagation"),
    ];

    let mut has_scaling = false;

    for (prefix, label) in &scaling_groups {
        let mut points: BTreeMap<u64, f64> = BTreeMap::new();
        for (name, values) in benchmarks.iter() {
            if let Some(suffix) = name.strip_prefix(&format!("{prefix}/")) {
                if let Ok(n) = suffix.parse::<u64>() {
                    let median = values["median"].as_f64().unwrap_or(0.0);
                    points.insert(n, median);
                }
            }
        }

        if points.len() >= 2 {
            if !has_scaling {
                report.push_str("## Scaling Analysis\n\n");
                has_scaling = true;
            }

            report.push_str(&format!("### {label}\n\n"));
            report.push_str("| Morphons | Time | Per-Morphon |\n");
            report.push_str("|----------|------|-------------|\n");

            for (&n, &time_ns) in &points {
                let per_morphon = time_ns / n as f64;
                report.push_str(&format!(
                    "| {} | {} | {} |\n",
                    n,
                    fmt_time(time_ns),
                    fmt_time(per_morphon),
                ));
            }

            // Compute scaling exponent: time ~ N^alpha
            // log(time) = alpha * log(N) + const
            if points.len() >= 2 {
                let log_points: Vec<(f64, f64)> = points
                    .iter()
                    .map(|(&n, &t)| ((n as f64).ln(), t.ln()))
                    .collect();

                let n = log_points.len() as f64;
                let sum_x: f64 = log_points.iter().map(|(x, _)| x).sum();
                let sum_y: f64 = log_points.iter().map(|(_, y)| y).sum();
                let sum_xy: f64 = log_points.iter().map(|(x, y)| x * y).sum();
                let sum_xx: f64 = log_points.iter().map(|(x, _)| x * x).sum();

                let denom = n * sum_xx - sum_x * sum_x;
                if denom.abs() > 1e-10 {
                    let alpha = (n * sum_xy - sum_x * sum_y) / denom;
                    let scaling_label = if alpha < 1.15 {
                        "linear"
                    } else if alpha < 1.5 {
                        "near-linear"
                    } else if alpha < 2.1 {
                        "quadratic"
                    } else {
                        "super-quadratic"
                    };
                    report.push_str(&format!(
                        "\nScaling exponent: **{alpha:.2}** ({scaling_label})\n"
                    ));
                }
            }
            report.push('\n');
        }
    }

    if !has_scaling {
        return;
    }
}

// === CartPole ===

fn write_cartpole_section(report: &mut String, results_dir: &str) {
    let path = format!("{results_dir}/cartpole_latest.json");
    let Some(data) = load_json(&path) else {
        report.push_str("## CartPole\n\n*No CartPole results found.*\n\n");
        return;
    };

    report.push_str("## CartPole\n\n");

    let results = &data["results"];
    let system = &data["system"];
    let diag = &data["diagnostics"];

    let solved = results["solved"].as_bool().unwrap_or(false);
    let status = if solved { "SOLVED" } else { "Not solved" };

    report.push_str("| Metric | Value |\n|--------|-------|\n");
    report.push_str(&format!("| **Status** | {} |\n", status));
    report.push_str(&format!("| Best Steps | {} |\n", results["best_steps"]));
    report.push_str(&format!("| Avg (last 100) | {:.1} |\n", results["avg_last_100"].as_f64().unwrap_or(0.0)));
    report.push_str(&format!("| Episodes | {} |\n", data["episodes"]));
    report.push_str(&format!("| Profile | {} |\n", data["profile"].as_str().unwrap_or("?")));

    report.push_str(&format!("\n**System State:** {} morphons, {} synapses, {} clusters, gen {}\n",
        system["morphons"], system["synapses"], system["clusters"], system["generation"]));
    report.push_str(&format!("**Firing Rate:** {:.3} | **Prediction Error:** {:.3}\n",
        system["firing_rate"].as_f64().unwrap_or(0.0),
        system["prediction_error"].as_f64().unwrap_or(0.0)));
    report.push_str(&format!("**Learning:** weight mean {:.3}, std {:.3}, {} active tags, {} captures\n\n",
        diag["weight_mean"].as_f64().unwrap_or(0.0),
        diag["weight_std"].as_f64().unwrap_or(0.0),
        diag["active_tags"],
        diag["total_captures"]));
}

// === Anomaly Detection ===

fn write_anomaly_section(report: &mut String, results_dir: &str) {
    let path = format!("{results_dir}/anomaly_latest.json");
    let Some(data) = load_json(&path) else {
        report.push_str("## Anomaly Detection\n\n*No anomaly results found.*\n\n");
        return;
    };

    report.push_str("## Anomaly Detection\n\n");

    report.push_str(&format!("Series length: {} | Anomaly rate: {:.0}% | Total anomalies: {} | Profile: {}\n\n",
        data["series_length"],
        data["anomaly_rate"].as_f64().unwrap_or(0.0) * 100.0,
        data["total_anomalies"],
        data["profile"].as_str().unwrap_or("?")));

    if let Some(results) = data["results"].as_array() {
        report.push_str("| Sigma | Threshold | Precision | Recall | F1 | TP | FP | FN |\n");
        report.push_str("|-------|-----------|-----------|--------|----|----|----|----|");
        report.push('\n');

        for r in results {
            report.push_str(&format!(
                "| {:.1} | {:.4} | {:.3} | {:.3} | {:.3} | {} | {} | {} |\n",
                r["sigma"].as_f64().unwrap_or(0.0),
                r["threshold"].as_f64().unwrap_or(0.0),
                r["precision"].as_f64().unwrap_or(0.0),
                r["recall"].as_f64().unwrap_or(0.0),
                r["f1"].as_f64().unwrap_or(0.0),
                r["tp"],
                r["fp"],
                r["fn"],
            ));
        }
    }

    let system = &data["system"];
    report.push_str(&format!("\n**System State:** {} morphons, {} synapses, firing rate {:.3}\n\n",
        system["morphons"], system["synapses"],
        system["firing_rate"].as_f64().unwrap_or(0.0)));
}

// === MNIST ===

fn write_mnist_section(report: &mut String, results_dir: &str) {
    let path = format!("{results_dir}/mnist_latest.json");
    let Some(data) = load_json(&path) else {
        report.push_str("## MNIST\n\n*No MNIST results found. Run with MNIST data in ./data/ to enable.*\n\n");
        return;
    };

    report.push_str("## MNIST\n\n");

    let results = &data["results"];
    report.push_str(&format!("**Test Accuracy: {:.1}%** | Train samples: {} | Test samples: {} | Epochs: {} | Profile: {}\n\n",
        results["test_accuracy"].as_f64().unwrap_or(0.0),
        data["train_samples"],
        data["test_samples"],
        data["epochs"],
        data["profile"].as_str().unwrap_or("?")));

    if let Some(per_class) = results["per_class"].as_array() {
        report.push_str("| Digit | Accuracy | Correct | Total |\n");
        report.push_str("|-------|----------|---------|-------|\n");

        for c in per_class {
            report.push_str(&format!(
                "| {} | {:.1}% | {} | {} |\n",
                c["digit"],
                c["accuracy"].as_f64().unwrap_or(0.0),
                c["correct"],
                c["total"],
            ));
        }
    }

    let system = &data["system"];
    report.push_str(&format!("\n**System State:** {} morphons, {} synapses, {} clusters\n\n",
        system["morphons"], system["synapses"], system["clusters"]));
}

// === Comparison vs Previous Version ===

fn write_comparison_section(report: &mut String, results_dir: &str) {
    let base = Path::new("docs/benchmark_results");
    let current_dir = format!("v{VERSION}");

    // Find previous version directory
    let prev_dir = find_previous_version(base, &current_dir);
    let Some(prev_dir) = prev_dir else { return };

    let prev_path = base.join(&prev_dir);

    report.push_str(&format!("## Comparison vs {prev_dir}\n\n"));

    // Compare CartPole
    let cur_cp = load_json(&format!("{results_dir}/cartpole_latest.json"));
    let prev_cp = load_json(&prev_path.join("cartpole_latest.json").to_string_lossy());

    if let (Some(cur), Some(prev)) = (&cur_cp, &prev_cp) {
        report.push_str("### CartPole\n\n");
        report.push_str("| Metric | Previous | Current | Delta |\n");
        report.push_str("|--------|----------|---------|-------|\n");

        let metrics = [
            ("Best Steps", "results.best_steps"),
            ("Avg (100)", "results.avg_last_100"),
            ("Morphons", "system.morphons"),
            ("Synapses", "system.synapses"),
        ];

        for (label, path) in &metrics {
            let parts: Vec<&str> = path.split('.').collect();
            let cur_val = cur[parts[0]][parts[1]].as_f64().unwrap_or(0.0);
            let prev_val = prev[parts[0]][parts[1]].as_f64().unwrap_or(0.0);
            report.push_str(&format!(
                "| {} | {:.1} | {:.1} | {} |\n",
                label, prev_val, cur_val, fmt_delta(cur_val, prev_val)
            ));
        }
        report.push('\n');
    }

    // Compare Criterion timings
    let cur_crit = load_json(&format!("{results_dir}/criterion_latest.json"));
    let prev_crit = load_json(&prev_path.join("criterion_latest.json").to_string_lossy());

    if let (Some(cur), Some(prev)) = (&cur_crit, &prev_crit) {
        if let (Some(cur_obj), Some(prev_obj)) = (cur.as_object(), prev.as_object()) {
            report.push_str("### Micro-Benchmarks\n\n");
            report.push_str("| Benchmark | Previous | Current | Delta |\n");
            report.push_str("|-----------|----------|---------|-------|\n");

            let mut keys: Vec<_> = cur_obj.keys().collect();
            keys.sort();

            for key in keys {
                let cur_median = cur_obj[key]["median"].as_f64().unwrap_or(0.0);
                if let Some(prev_val) = prev_obj.get(key) {
                    let prev_median = prev_val["median"].as_f64().unwrap_or(0.0);
                    report.push_str(&format!(
                        "| {} | {} | {} | {} |\n",
                        key,
                        fmt_time(prev_median),
                        fmt_time(cur_median),
                        fmt_delta(cur_median, prev_median),
                    ));
                }
            }
            report.push('\n');
        }
    }
}

fn find_previous_version(base: &Path, current: &str) -> Option<String> {
    let mut versions: Vec<String> = fs::read_dir(base)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with('v') && name != current)
        .collect();

    versions.sort();
    versions.pop() // highest version below current (simple sort works for semver)
}

// === Footer ===

fn write_footer(report: &mut String) {
    report.push_str("---\n\n");
    report.push_str(&format!("*Generated by `bench_report` v{VERSION}*\n"));
}

// === HTML Dashboard ===

fn generate_html_dashboard(results_dir: &str, profile: &str) -> String {
    let now = chrono_lite();

    // Load all data
    let criterion = load_json(&format!("{results_dir}/criterion_latest.json"));
    let cartpole = load_json(&format!("{results_dir}/cartpole_latest.json"));
    let anomaly = load_json(&format!("{results_dir}/anomaly_latest.json"));
    let mnist = load_json(&format!("{results_dir}/mnist_latest.json"));

    // Build chart data
    let (micro_labels, micro_values) = build_micro_chart_data(&criterion);
    let (scaling_labels, scaling_step, scaling_resonance) = build_scaling_chart_data(&criterion);
    let (anomaly_sigmas, anomaly_p, anomaly_r, anomaly_f1) = build_anomaly_chart_data(&anomaly);
    let (mnist_digits, mnist_acc) = build_mnist_chart_data(&mnist);

    // CartPole stats
    let cp_best = cartpole.as_ref().map(|d| d["results"]["best_steps"].as_u64().unwrap_or(0)).unwrap_or(0);
    let cp_avg = cartpole.as_ref().map(|d| d["results"]["avg_last_100"].as_f64().unwrap_or(0.0)).unwrap_or(0.0);
    let cp_solved = cartpole.as_ref().map(|d| d["results"]["solved"].as_bool().unwrap_or(false)).unwrap_or(false);
    let cp_morphons = cartpole.as_ref().map(|d| d["system"]["morphons"].as_u64().unwrap_or(0)).unwrap_or(0);
    let cp_synapses = cartpole.as_ref().map(|d| d["system"]["synapses"].as_u64().unwrap_or(0)).unwrap_or(0);
    let cp_clusters = cartpole.as_ref().map(|d| d["system"]["clusters"].as_u64().unwrap_or(0)).unwrap_or(0);
    let cp_fr = cartpole.as_ref().map(|d| d["system"]["firing_rate"].as_f64().unwrap_or(0.0)).unwrap_or(0.0);

    // Anomaly system
    let an_morphons = anomaly.as_ref().map(|d| d["system"]["morphons"].as_u64().unwrap_or(0)).unwrap_or(0);
    let an_series = anomaly.as_ref().map(|d| d["series_length"].as_u64().unwrap_or(0)).unwrap_or(0);

    // MNIST
    let mn_acc = mnist.as_ref().map(|d| d["results"]["test_accuracy"].as_f64().unwrap_or(0.0)).unwrap_or(0.0);
    let mn_morphons = mnist.as_ref().map(|d| d["system"]["morphons"].as_u64().unwrap_or(0)).unwrap_or(0);
    let mn_synapses = mnist.as_ref().map(|d| d["system"]["synapses"].as_u64().unwrap_or(0)).unwrap_or(0);

    // Scaling exponent
    let scaling_exp = compute_scaling_exponent(&criterion, "system_step_scaling");
    let resonance_exp = compute_scaling_exponent(&criterion, "resonance_propagate");

    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Morphon Benchmark Dashboard v{VERSION}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --text: #e4e4e7;
    --muted: #71717a;
    --accent: #818cf8;
    --accent2: #34d399;
    --accent3: #f472b6;
    --accent4: #fbbf24;
    --danger: #ef4444;
    --success: #22c55e;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.6;
  }}
  .header {{
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
  }}
  .header .meta {{
    color: var(--muted);
    font-size: 0.85rem;
  }}
  .header .meta span {{ margin: 0 0.75rem; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: var(--accent); }}
  .card h2 {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
  }}
  .card.wide {{ grid-column: 1 / -1; }}
  .stat-row {{
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }}
  .stat {{
    flex: 1;
    min-width: 100px;
    text-align: center;
    padding: 0.75rem;
    background: var(--bg);
    border-radius: 8px;
  }}
  .stat .value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
  }}
  .stat .value.success {{ color: var(--success); }}
  .stat .value.danger {{ color: var(--danger); }}
  .stat .value.warn {{ color: var(--accent4); }}
  .stat .label {{
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.25rem;
  }}
  .chart-container {{
    position: relative;
    height: 280px;
  }}
  .chart-container.tall {{
    height: 350px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th, td {{
    padding: 0.5rem 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    color: var(--muted);
    font-weight: 500;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
  }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .badge {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge.solved {{ background: rgba(34, 197, 94, 0.15); color: var(--success); }}
  .badge.unsolved {{ background: rgba(239, 68, 68, 0.15); color: var(--danger); }}
  footer {{
    text-align: center;
    color: var(--muted);
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<div class="header">
  <h1>MORPHON</h1>
  <div class="meta">
    <span>v{VERSION}</span>
    <span>|</span>
    <span>{profile}</span>
    <span>|</span>
    <span>{now}</span>
    <span>|</span>
    <span>{os} {arch}</span>
  </div>
</div>

<!-- CartPole -->
<div class="grid">
  <div class="card">
    <h2>CartPole Control</h2>
    <div class="stat-row">
      <div class="stat">
        <div class="value {cp_solved_class}">{cp_best}</div>
        <div class="label">Best Steps</div>
      </div>
      <div class="stat">
        <div class="value">{cp_avg:.1}</div>
        <div class="label">Avg (100)</div>
      </div>
      <div class="stat">
        <div class="value">{cp_morphons}</div>
        <div class="label">Morphons</div>
      </div>
      <div class="stat">
        <div class="value">{cp_synapses}</div>
        <div class="label">Synapses</div>
      </div>
    </div>
    <div class="stat-row">
      <div class="stat">
        <div class="value">{cp_clusters}</div>
        <div class="label">Clusters</div>
      </div>
      <div class="stat">
        <div class="value">{cp_fr:.3}</div>
        <div class="label">Firing Rate</div>
      </div>
      <div class="stat">
        <span class="badge {cp_badge}">{cp_status}</span>
        <div class="label" style="margin-top:0.5rem">Status</div>
      </div>
    </div>
  </div>

  <!-- Anomaly Detection -->
  <div class="card">
    <h2>Anomaly Detection</h2>
    <div class="chart-container">
      <canvas id="anomalyChart"></canvas>
    </div>
    <div class="stat-row" style="margin-top:0.75rem">
      <div class="stat">
        <div class="value">{an_morphons}</div>
        <div class="label">Morphons</div>
      </div>
      <div class="stat">
        <div class="value">{an_series}</div>
        <div class="label">Series Len</div>
      </div>
    </div>
  </div>
</div>

<!-- Micro-Benchmarks -->
<div class="grid">
  <div class="card wide">
    <h2>Micro-Benchmarks (Criterion)</h2>
    <div class="chart-container tall">
      <canvas id="microChart"></canvas>
    </div>
  </div>
</div>

<!-- Scaling -->
<div class="grid">
  <div class="card">
    <h2>System Step Scaling</h2>
    <div class="chart-container">
      <canvas id="scalingStepChart"></canvas>
    </div>
    <div style="text-align:center;margin-top:0.5rem;color:var(--muted);font-size:0.8rem">
      Scaling exponent: <span style="color:var(--accent)">{scaling_exp:.2}</span>
    </div>
  </div>
  <div class="card">
    <h2>Resonance Propagation Scaling</h2>
    <div class="chart-container">
      <canvas id="scalingResChart"></canvas>
    </div>
    <div style="text-align:center;margin-top:0.5rem;color:var(--muted);font-size:0.8rem">
      Scaling exponent: <span style="color:var(--accent)">{resonance_exp:.2}</span>
    </div>
  </div>
</div>

<!-- MNIST -->
<div class="grid">
  <div class="card">
    <h2>MNIST Classification</h2>
    <div class="stat-row">
      <div class="stat">
        <div class="value {mn_acc_class}">{mn_acc:.1}%</div>
        <div class="label">Test Accuracy</div>
      </div>
      <div class="stat">
        <div class="value">{mn_morphons}</div>
        <div class="label">Morphons</div>
      </div>
      <div class="stat">
        <div class="value">{mn_synapses}</div>
        <div class="label">Synapses</div>
      </div>
    </div>
    <div class="chart-container" style="height:200px">
      <canvas id="mnistChart"></canvas>
    </div>
  </div>
</div>

<footer>
  Generated by bench_report v{VERSION}
</footer>

<script>
Chart.defaults.color = '#71717a';
Chart.defaults.borderColor = '#2a2d3a';
Chart.defaults.font.family = "'SF Mono', 'Fira Code', monospace";
Chart.defaults.font.size = 11;

// Micro-benchmarks (horizontal bar, log scale)
new Chart(document.getElementById('microChart'), {{
  type: 'bar',
  data: {{
    labels: {micro_labels},
    datasets: [{{
      label: 'Median (us)',
      data: {micro_values},
      backgroundColor: 'rgba(129, 140, 248, 0.6)',
      borderColor: 'rgba(129, 140, 248, 1)',
      borderWidth: 1,
      borderRadius: 4,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{
        type: 'logarithmic',
        title: {{ display: true, text: 'Time (us)' }},
        grid: {{ color: '#1e2030' }},
      }},
      y: {{ grid: {{ display: false }} }}
    }}
  }}
}});

// Scaling: System Step
new Chart(document.getElementById('scalingStepChart'), {{
  type: 'line',
  data: {{
    labels: {scaling_labels},
    datasets: [{{
      label: 'System Step',
      data: {scaling_step},
      borderColor: '#818cf8',
      backgroundColor: 'rgba(129, 140, 248, 0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 5,
      pointHoverRadius: 7,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Morphons' }}, grid: {{ color: '#1e2030' }} }},
      y: {{ title: {{ display: true, text: 'Time (us)' }}, grid: {{ color: '#1e2030' }} }}
    }}
  }}
}});

// Scaling: Resonance
new Chart(document.getElementById('scalingResChart'), {{
  type: 'line',
  data: {{
    labels: {scaling_labels},
    datasets: [{{
      label: 'Resonance',
      data: {scaling_resonance},
      borderColor: '#34d399',
      backgroundColor: 'rgba(52, 211, 153, 0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 5,
      pointHoverRadius: 7,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Morphons' }}, grid: {{ color: '#1e2030' }} }},
      y: {{ title: {{ display: true, text: 'Time (us)' }}, grid: {{ color: '#1e2030' }} }}
    }}
  }}
}});

// Anomaly Detection
new Chart(document.getElementById('anomalyChart'), {{
  type: 'bar',
  data: {{
    labels: {anomaly_sigmas},
    datasets: [
      {{ label: 'Precision', data: {anomaly_p}, backgroundColor: 'rgba(129, 140, 248, 0.7)', borderRadius: 4 }},
      {{ label: 'Recall', data: {anomaly_r}, backgroundColor: 'rgba(52, 211, 153, 0.7)', borderRadius: 4 }},
      {{ label: 'F1', data: {anomaly_f1}, backgroundColor: 'rgba(244, 114, 182, 0.7)', borderRadius: 4 }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Sigma Threshold' }}, grid: {{ display: false }} }},
      y: {{ title: {{ display: true, text: 'Score' }}, min: 0, max: 1, grid: {{ color: '#1e2030' }} }}
    }}
  }}
}});

// MNIST per-digit
new Chart(document.getElementById('mnistChart'), {{
  type: 'bar',
  data: {{
    labels: {mnist_digits},
    datasets: [{{
      label: 'Accuracy %',
      data: {mnist_acc},
      backgroundColor: function(ctx) {{
        const v = ctx.raw;
        if (v > 30) return 'rgba(52, 211, 153, 0.7)';
        if (v > 10) return 'rgba(251, 191, 36, 0.7)';
        return 'rgba(239, 68, 68, 0.5)';
      }},
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Digit' }}, grid: {{ display: false }} }},
      y: {{ title: {{ display: true, text: 'Accuracy %' }}, min: 0, max: 100, grid: {{ color: '#1e2030' }} }}
    }}
  }}
}});
</script>
</body>
</html>"##,
        VERSION = VERSION,
        profile = profile,
        now = now,
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        cp_best = cp_best,
        cp_avg = cp_avg,
        cp_morphons = cp_morphons,
        cp_synapses = cp_synapses,
        cp_clusters = cp_clusters,
        cp_fr = cp_fr,
        cp_solved_class = if cp_solved { "success" } else { "" },
        cp_badge = if cp_solved { "solved" } else { "unsolved" },
        cp_status = if cp_solved { "SOLVED" } else { "NOT SOLVED" },
        an_morphons = an_morphons,
        an_series = an_series,
        mn_acc = mn_acc,
        mn_acc_class = if mn_acc > 50.0 { "success" } else if mn_acc > 20.0 { "warn" } else { "danger" },
        mn_morphons = mn_morphons,
        mn_synapses = mn_synapses,
        micro_labels = micro_labels,
        micro_values = micro_values,
        scaling_labels = scaling_labels,
        scaling_step = scaling_step,
        scaling_resonance = scaling_resonance,
        scaling_exp = scaling_exp,
        resonance_exp = resonance_exp,
        anomaly_sigmas = anomaly_sigmas,
        anomaly_p = anomaly_p,
        anomaly_r = anomaly_r,
        anomaly_f1 = anomaly_f1,
        mnist_digits = mnist_digits,
        mnist_acc = mnist_acc,
    )
}

fn build_micro_chart_data(criterion: &Option<Value>) -> (String, String) {
    let Some(data) = criterion.as_ref().and_then(|d| d.as_object()) else {
        return ("[]".into(), "[]".into());
    };

    let mut entries: Vec<_> = data.iter()
        .filter(|(k, _)| !k.contains('/')) // exclude grouped benchmarks
        .collect();
    entries.sort_by_key(|(k, _)| k.to_string());

    let labels: Vec<String> = entries.iter().map(|(k, _)| format!("\"{}\"", k)).collect();
    let values: Vec<String> = entries.iter()
        .map(|(_, v)| format!("{:.2}", v["median"].as_f64().unwrap_or(0.0) / 1_000.0))
        .collect();

    (format!("[{}]", labels.join(",")), format!("[{}]", values.join(",")))
}

fn build_scaling_chart_data(criterion: &Option<Value>) -> (String, String, String) {
    let Some(data) = criterion.as_ref().and_then(|d| d.as_object()) else {
        return ("[]".into(), "[]".into(), "[]".into());
    };

    let mut step_points: BTreeMap<u64, f64> = BTreeMap::new();
    let mut res_points: BTreeMap<u64, f64> = BTreeMap::new();

    for (name, values) in data.iter() {
        let median_us = values["median"].as_f64().unwrap_or(0.0) / 1_000.0;
        if let Some(n) = name.strip_prefix("system_step_scaling/") {
            if let Ok(n) = n.parse::<u64>() { step_points.insert(n, median_us); }
        }
        if let Some(n) = name.strip_prefix("resonance_propagate/") {
            if let Ok(n) = n.parse::<u64>() { res_points.insert(n, median_us); }
        }
    }

    let labels: Vec<String> = step_points.keys().map(|k| k.to_string()).collect();
    let step_vals: Vec<String> = step_points.values().map(|v| format!("{v:.1}")).collect();
    let res_vals: Vec<String> = labels.iter()
        .filter_map(|l| l.parse::<u64>().ok())
        .map(|n| res_points.get(&n).map(|v| format!("{v:.1}")).unwrap_or("null".into()))
        .collect();

    (
        format!("[{}]", labels.join(",")),
        format!("[{}]", step_vals.join(",")),
        format!("[{}]", res_vals.join(",")),
    )
}

fn build_anomaly_chart_data(anomaly: &Option<Value>) -> (String, String, String, String) {
    let empty = || ("[]".into(), "[]".into(), "[]".into(), "[]".into());
    let Some(data) = anomaly else { return empty(); };
    let Some(results) = data["results"].as_array() else { return empty(); };

    let sigmas: Vec<String> = results.iter()
        .map(|r| format!("\"{}\"", r["sigma"].as_f64().unwrap_or(0.0)))
        .collect();
    let p: Vec<String> = results.iter().map(|r| format!("{:.3}", r["precision"].as_f64().unwrap_or(0.0))).collect();
    let r: Vec<String> = results.iter().map(|r| format!("{:.3}", r["recall"].as_f64().unwrap_or(0.0))).collect();
    let f1: Vec<String> = results.iter().map(|r| format!("{:.3}", r["f1"].as_f64().unwrap_or(0.0))).collect();

    (
        format!("[{}]", sigmas.join(",")),
        format!("[{}]", p.join(",")),
        format!("[{}]", r.join(",")),
        format!("[{}]", f1.join(",")),
    )
}

fn build_mnist_chart_data(mnist: &Option<Value>) -> (String, String) {
    let Some(data) = mnist else { return ("[]".into(), "[]".into()); };
    let Some(per_class) = data["results"]["per_class"].as_array() else {
        return ("[]".into(), "[]".into());
    };

    let digits: Vec<String> = per_class.iter().map(|c| format!("{}", c["digit"])).collect();
    let acc: Vec<String> = per_class.iter()
        .map(|c| format!("{:.1}", c["accuracy"].as_f64().unwrap_or(0.0)))
        .collect();

    (format!("[{}]", digits.join(",")), format!("[{}]", acc.join(",")))
}

fn compute_scaling_exponent(criterion: &Option<Value>, prefix: &str) -> f64 {
    let Some(data) = criterion.as_ref().and_then(|d| d.as_object()) else { return 0.0 };

    let points: Vec<(f64, f64)> = data.iter()
        .filter_map(|(name, values)| {
            let suffix = name.strip_prefix(&format!("{prefix}/"))?;
            let n: f64 = suffix.parse().ok()?;
            let t = values["median"].as_f64()?;
            Some((n.ln(), t.ln()))
        })
        .collect();

    if points.len() < 2 { return 0.0; }

    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 { return 0.0; }
    (n * sum_xy - sum_x * sum_y) / denom
}
