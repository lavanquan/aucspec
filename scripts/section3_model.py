"""AUC_SPEC Section 3: fit the verifier roofline and evaluate the symmetric
reference model against measured capacity.

  fit      eqs. (35)-(37) from verify-call measurements, either the offline
           profile written by legacy/scripts/profile_target.py or the per-batch
           rows of simulator rounds CSVs (--detailed-log).
  predict  N_sym(x), per-depth N_gamma(x), gamma* (eq. 51), N_knee(gamma)
           (eq. 39) and gamma* transitions from a fit plus deployment
           parameters; optionally side by side with a measured
           nstar_frontier_<policy>.csv.

Examples:
  python scripts/section3_model.py fit --profile-csv results/target_profile.csv \
      --output results/section3/roofline_fit.json
  python scripts/section3_model.py fit --rounds-csv RUN1.csv RUN2.csv \
      --output results/section3/roofline_fit_rounds.json
  python scripts/section3_model.py predict --fit results/section3/roofline_fit.json \
      --system configs/section3_system.yaml --output results/section3/nsym.csv \
      [--measured results/capacity_frontier/budget100_x*/nstar_frontier_capacity_dpp.csv]

The roofline fit only identifies the compute regime if the measurements
reach it: re-profile with large batches (e.g. profile_target.py defaults,
B up to 32, gamma up to 16, context up to 4096) -- the frontier runs so far
stayed at B <= 7 and Gamma <= 31 (CANONICAL_EXPERIMENT_PREFLIGHT.md 3.1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))

from edge_specsim.capacity_model import (  # noqa: E402
    RooflineParams,
    SymmetricSystem,
    fit_roofline,
)


def _token_list_length(value) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    text = str(value).strip().strip("[]").strip()
    return 0 if not text else text.count(",") + 1


def _profile_points(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "batch_size": df["batch_size"].astype(float),
        "verifier_tokens": df["total_verifier_tokens"].astype(float),
        "context_tokens": (df["batch_size"] * df["context_length"]).astype(float),
        "wall_ms": df["wall_time_ms"].astype(float),
    })


def _rounds_points(paths: list[str]) -> pd.DataFrame:
    cols = ["verification_batch_id", "verification_batch_size", "verification_batch_token_cost",
            "verification_batch_measured_service_ms", "context_token_ids"]
    frames = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
        missing = set(cols) - set(df.columns)
        if missing:
            raise SystemExit(f"{path} lacks {sorted(missing)} (run with --detailed-log)")
        df["ctx"] = df["context_token_ids"].map(_token_list_length)
        per_batch = df.groupby("verification_batch_id").agg(
            batch_size=("verification_batch_size", "first"),
            verifier_tokens=("verification_batch_token_cost", "first"),
            context_tokens=("ctx", "sum"),
            wall_ms=("verification_batch_measured_service_ms", "first"),
        )
        per_batch["source"] = i
        frames.append(per_batch.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def cmd_fit(args: argparse.Namespace) -> None:
    if bool(args.profile_csv) == bool(args.rounds_csv):
        raise SystemExit("give exactly one of --profile-csv / --rounds-csv")
    pts = _profile_points(args.profile_csv) if args.profile_csv else _rounds_points(args.rounds_csv)
    fit = fit_roofline(pts["batch_size"], pts["verifier_tokens"], pts["context_tokens"], pts["wall_ms"])
    fit["source"] = args.profile_csv or args.rounds_csv
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(fit, indent=2), encoding="utf-8")
    print(json.dumps(fit, indent=2))
    if not fit["compute_regime_sampled"]:
        print("!! fewer than 3 compute-bound points: theta_f and the knee are not identified "
              "by this data -- profile larger batches / deeper gamma", flush=True)


def _system(fit: dict, sys_cfg: dict) -> SymmetricSystem:
    roofline = RooflineParams(
        theta_0_ms=fit["theta_0_ms"], a_mem_ms=fit["a_mem_ms"],
        kv_ms_per_ctx=fit["kv_ms_per_ctx"], theta_f_ms=fit["theta_f_ms"],
    )
    return SymmetricSystem(
        alpha=float(sys_cfg["alpha"]),
        gamma_choices=[int(g) for g in sys_cfg["gamma_choices"]],
        roofline=roofline,
        context_length=float(sys_cfg["context_length"]),
        draft_ms_per_token=float(sys_cfg.get("draft_ms_per_token", 0.0)),
        draft_fixed_ms=float(sys_cfg.get("draft_fixed_ms", 0.0)),
        rtt_ms=float(sys_cfg.get("rtt_ms", 0.0)),
        kappa_bits=float(sys_cfg.get("kappa_bits", 0.0)),
        bandwidth_hz=float(sys_cfg.get("bandwidth_hz", float("inf"))),
        spectral_efficiency=float(sys_cfg.get("spectral_efficiency", 1.0)),
    )


def cmd_predict(args: argparse.Namespace) -> None:
    fit = json.loads(Path(args.fit).read_text(encoding="utf-8"))
    sys_cfg = yaml.safe_load(Path(args.system).read_text(encoding="utf-8"))
    system = _system(fit, sys_cfg)
    n_max = int(sys_cfg.get("n_max", 64))

    rows = []
    for x in [float(v) for v in sys_cfg["x_grid"]]:
        row = {"x": x, "N_sym": system.n_sym(x, n_max)}
        best_n, best_g = -1, None
        for g in system.gamma_choices:
            n_g = max((n for n in range(1, n_max + 1) if system.x_gamma(n, g) >= x), default=0)
            row[f"N_gamma{g}"] = n_g
            if n_g > best_n:
                best_n, best_g = n_g, g
        row["gamma_star"] = best_g
        rows.append(row)
    out = pd.DataFrame(rows)

    if args.measured:
        meas = pd.concat([pd.read_csv(m) for m in args.measured], ignore_index=True)
        meas = meas[["x_requirement", "N_lower", "N_upper", "exact_boundary"]]
        meas = meas.rename(columns={"x_requirement": "x", "N_lower": "N_measured_lower",
                                    "N_upper": "N_measured_upper"})
        out = out.merge(meas, on="x", how="outer").sort_values("x")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    knees = {int(g): system.n_knee(g) for g in system.gamma_choices}
    summary = {
        "n_knee": knees,
        "gamma_transitions": system.gamma_transitions(n_max),
        "x_sym": [{"n": n, "x": system.x_sym(n)[0], "gamma": system.x_sym(n)[1],
                   "regime": system.roofline.regime(n, system.x_sym(n)[1], system.context_length)}
                  for n in range(1, n_max + 1)],
    }
    summary_path = Path(args.output).with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nN_knee(gamma): {knees}")
    print(f"gamma* transitions: {summary['gamma_transitions']}")
    print(f"wrote {args.output} and {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--profile-csv")
    f.add_argument("--rounds-csv", nargs="+")
    f.add_argument("--output", required=True)
    f.set_defaults(func=cmd_fit)
    q = sub.add_parser("predict")
    q.add_argument("--fit", required=True)
    q.add_argument("--system", required=True, help="YAML with alpha, gamma_choices, context_length, ... x_grid")
    q.add_argument("--measured", nargs="+",
                   help="nstar_frontier_<policy>.csv file(s) to compare against (one per x is fine)")
    q.add_argument("--output", required=True)
    q.set_defaults(func=cmd_predict)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
