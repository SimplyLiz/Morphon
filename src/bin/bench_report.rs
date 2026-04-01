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
