"""Decisions, boundary verdict, and report for the budget100 x=3.0 certification.

No GPU. Reads <run_dir>/cache_parts/*.jsonl, applies the existing exact-SLO
final classifier (classify_feasibility_final) to the shortest seed prefix
k in [initial_seeds, max_seeds] that gives a decisive verdict, and writes the
compact outputs. A candidate with any exhausted client is INVALID.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))

from edge_specsim.capacity import CandidateCache, classify_feasibility_final  # noqa: E402

RESULTS_ROOT = REPO_ROOT / "results" / "capacity_frontier"


def decide(by_seed: dict, x: float, n: int, initial: int, max_seeds: int, delta: float) -> dict:
    """by_seed: {seed_id: CandidateMeasurement}. Seeds must be contiguous from 1."""
    contiguous = []
    for s in range(1, max_seeds + 1):
        if s not in by_seed:
            break
        contiguous.append(s)
    out = {"n": n, "seeds_available": contiguous, "x": x}
    exhausted = [s for s in contiguous if by_seed[s].extra.get("exhausted_clients")]
    if exhausted:
        out.update(status="invalid_exhausted", seeds_used=[], invalid_seeds=exhausted)
        return out
    if len(contiguous) < initial:
        out.update(status="incomplete", seeds_used=[])
        return out
    last = None
    for k in range(initial, len(contiguous) + 1):
        last = classify_feasibility_final([by_seed[s] for s in contiguous[:k]], x, n, confidence_delta=delta)
        if last.status in ("feasible", "infeasible"):
            out.update(status=last.status, seeds_used=contiguous[:k], min_lcb=last.min_lcb, min_ucb=last.min_ucb)
            break
    else:
        out.update(
            status="uncertain_at_max_seeds" if len(contiguous) >= max_seeds else "needs_more_seeds",
            seeds_used=contiguous, min_lcb=last.min_lcb, min_ucb=last.min_ucb,
        )
    ms = [by_seed[s] for s in out["seeds_used"]]
    rates = [m.min_rate_tps for m in ms]
    out.update(
        min_rate_min=min(rates), min_rate_max=max(rates), min_rate_mean=statistics.fmean(rates),
        mean_gamma=statistics.fmean(m.mean_gamma for m in ms),
        mean_batch=statistics.fmean(m.mean_batch_size for m in ms),
        mean_fill=statistics.fmean(m.mean_fill_ratio for m in ms),
        goodput=statistics.fmean(m.goodput_tps for m in ms),
    )
    return out


def conclude(dec: dict) -> tuple[str, str]:
    """dec: {N: decision dict}. Returns (kind, text)."""
    def st(n):
        return dec.get(n, {}).get("status", "not_run")

    if st(10) == "invalid_exhausted":
        return "invalid", "N*(3.0) unresolved: N=10 runs invalid (workload exhaustion); rerun with larger workload supply."
    if st(10) == "infeasible":
        if st(9) == "feasible":
            return "certified", "N*(3.0) = 9, certified"
        return "unresolved", f"N*(3.0) unresolved: N=10 infeasible but N=9 status is {st(9)} under the 480 s protocol."
    if st(10) == "feasible":
        if st(11) == "infeasible":
            return "certified", "N*(3.0) = 10, certified"
        if st(11) == "feasible":
            if st(12) == "infeasible":
                return "certified", "N*(3.0) = 11, certified"
            if st(12) == "feasible":
                return "anomaly", "N*(3.0) >= 12 under 480 s: contradicts the 240 s sweep (N=12 infeasible). Search upward."
            return "unresolved", f"N*(3.0) unresolved: N=10, 11 feasible; N=12 status is {st(12)}."
        return "unresolved", f"N*(3.0) unresolved: N=10 feasible; N=11 status is {st(11)}."
    if st(10) == "uncertain_at_max_seeds":
        return "unresolved", "N*(3.0) in [9, 11], unresolved (N=10 uncertain after the maximum number of 480 s seeds; 9 and 12 bounds come from the earlier 240 s sweep)"
    return "pending", f"N=10 status is {st(10)}; certification not finished."


def load(run_dir: Path) -> tuple[dict, dict]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    per_n: dict[int, dict] = {}
    for part in sorted((run_dir / "cache_parts").glob("n*_s*.jsonl")):
        c = CandidateCache(part)
        for m in c._mem.values():
            per_n.setdefault(m.n_active, {})[m.seed] = m
    return cfg, per_n


def build(run_dir: Path) -> dict:
    cfg, per_n = load(run_dir)
    cs = cfg["capacity_search"]
    x = float(cs["x_grid_final"][0])
    decs = {
        n: decide(by_seed, x, n, int(cs["initial_seeds"]), int(cs["max_seeds"]), float(cs["confidence_delta"]))
        for n, by_seed in sorted(per_n.items())
    }
    kind, text = conclude(decs)
    return {"cfg": cfg, "per_n": per_n, "decs": decs, "kind": kind, "text": text, "x": x}


def write_outputs(run_dir: Path) -> dict:
    r = build(run_dir)
    cs = r["cfg"]["capacity_search"]
    meas_s = float(cs["measurement_seconds_final"])
    # compact merged cache (schema-compatible records, one per candidate/seed)
    with open(run_dir / "candidate_cache.jsonl", "w", encoding="utf-8") as out:
        for part in sorted((run_dir / "cache_parts").glob("n*_s*.jsonl")):
            out.write(part.read_text(encoding="utf-8"))
    with open(run_dir / "candidate_seed_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "seed", "measurement_s", "min_rate_tps", "goodput_tps", "mean_gamma",
                    "mean_batch_size", "mean_fill_ratio", "max_deficit_queue_slope",
                    "server_queue_slope", "max_device_queue_slope", "forced_service_fraction",
                    "exhausted_clients", "per_client_rates"])
        for n, by_seed in sorted(r["per_n"].items()):
            for s, m in sorted(by_seed.items()):
                w.writerow([n, s, m.extra.get("measurement_duration_s"), m.min_rate_tps, m.goodput_tps,
                            m.mean_gamma, m.mean_batch_size, m.mean_fill_ratio, m.max_z_slope,
                            m.server_queue_slope, m.max_device_queue_slope,
                            m.extra.get("forced_service_fraction"),
                            json.dumps(m.extra.get("exhausted_clients") or []),
                            json.dumps({str(k): round(v, 4) for k, v in m.per_client_rates.items()})])
    with open(run_dir / "nstar_boundary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "N", "status", "seeds_used", "measurement_s", "min_rate_min", "min_rate_max",
                    "min_rate_mean", "min_lcb", "min_ucb", "mean_gamma", "mean_batch", "goodput"])
        for n, d in sorted(r["decs"].items()):
            w.writerow([r["x"], n, d["status"], len(d.get("seeds_used", [])), meas_s,
                        d.get("min_rate_min"), d.get("min_rate_max"), d.get("min_rate_mean"),
                        d.get("min_lcb"), d.get("min_ucb"), d.get("mean_gamma"), d.get("mean_batch"),
                        d.get("goodput")])
    return r


def fmt(v, p=3):
    return "-" if v is None else f"{v:.{p}f}"


def write_report(run_dir: Path, out_md: Path) -> None:
    r = write_outputs(run_dir)
    cs = r["cfg"]["capacity_search"]
    meas_s = float(cs["measurement_seconds_final"])
    git = (run_dir / "git_commit.txt").read_text().strip()
    chash = (run_dir / "config_hash.txt").read_text().strip()
    fp = (run_dir / "population_fingerprint.txt").read_text().strip()
    exhausted_any = any(
        m.extra.get("exhausted_clients") for by in r["per_n"].values() for m in by.values()
    )
    L = []
    L.append("# budget100, x=3.0 boundary certification (capacity_dpp)\n")
    L.append(f"**Conclusion: {r['text']}**\n")
    L.append("| N | status | seeds | measurement_s | min-rate range | min-rate mean | CI/decision | mean gamma | mean batch | goodput |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n, d in sorted(r["decs"].items()):
        k = len(d.get("seeds_used", []))
        rng = "-" if d.get("min_rate_min") is None else f"{d['min_rate_min']:.3f}-{d['min_rate_max']:.3f}"
        ci = "-" if d.get("min_lcb") is None else f"LCB {d['min_lcb']:.3f} / UCB {d['min_ucb']:.3f} vs x=3.0"
        L.append(f"| {n} | {d['status']} | {k} | {meas_s:.0f} | {rng} | {fmt(d.get('min_rate_mean'))} | {ci} | "
                 f"{fmt(d.get('mean_gamma'), 2)} | {fmt(d.get('mean_batch'), 2)} | {fmt(d.get('goodput'), 2)} |")
    L.append("")
    L.append("Decision rule: shortest seed prefix k in [%d, %d] whose exact-SLO Bonferroni-bootstrap bounds "
             "(delta=%s) are decisive; feasible iff min_i LCB >= 3.0, infeasible iff min_i UCB < 3.0, else uncertain."
             % (cs["initial_seeds"], cs["max_seeds"], cs["confidence_delta"]))
    L.append("")
    L.append("## Provenance\n")
    L.append(f"- git commit used for runs: `{git}`")
    L.append(f"- config hash: `{chash}` (includes measurement_seconds_final=%s, so 240 s records cannot be reused)" % meas_s)
    L.append(f"- population fingerprint: `{fp}`")
    L.append(f"- warmup_s={cs['warmup_seconds']}, measurement_s={meas_s}, initial_seeds={cs['initial_seeds']}, max_seeds={cs['max_seeds']}")
    for n, d in sorted(r["decs"].items()):
        L.append(f"- N={n}: seeds run {d['seeds_available']}; seeds used for verdict {d.get('seeds_used', [])}")
    L.append(f"- workload exhaustion occurred: {'YES' if exhausted_any else 'no'}")
    unc = [n for n, d in r["decs"].items() if d["status"] in ("uncertain_at_max_seeds", "needs_more_seeds")]
    L.append(f"- candidates still uncertain: {unc if unc else 'none'}")
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["status", "report"])
    p.add_argument("--run-tag", required=True)
    a = p.parse_args()
    run_dir = RESULTS_ROOT / a.run_tag
    if a.cmd == "status":
        r = build(run_dir)
        for n, d in sorted(r["decs"].items()):
            print(n, d["status"], "avail", d["seeds_available"], "used", d.get("seeds_used"),
                  "lcb", fmt(d.get("min_lcb")), "ucb", fmt(d.get("min_ucb")))
        print(r["kind"], "|", r["text"])
    else:
        write_report(run_dir, REPO_ROOT / "results" / "BUDGET100_X3_CERTIFICATION_REPORT.md")
        print("report written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
