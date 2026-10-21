"""C3.1 Task A: measured workload conditions in the valid C1.3/C1.4 GPU traces vs the assumptions of the C3 CPU scenarios.
CPU only; reads the compact replay tables committed in C2 (results/c2/*.csv.gz).  Anything the traces cannot identify is reported UNKNOWN.
usage: trace_conditions.py <out.json>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS_TRAIN = ["RQ1_CAP16_5a37f03", "RQ1_EXT_28907a6", "RQ1_PROBE_2d58666"]      # GSM8K-train, gamma = 2, FCFS (ext = cap-32 serving, N<=16 so the cap never bound)


def q(x, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
    x = np.asarray(x, float)
    return {f"p{int(p * 100)}": float(np.quantile(x, p)) for p in ps} if len(x) else {}


def censored_alpha(acc: np.ndarray, gamma: int):
    """Per-token acceptance under a geometric model with right censoring at gamma: alpha = S / (S + F), S = accepted tokens, F = observed rejections."""
    s = float(acc.sum())
    f = float((acc < gamma).sum())
    return (s / (s + f) if (s + f) > 0 else float("nan")), s + f


def main(out):
    r = pd.read_csv(ROOT / "results" / "c2" / "rounds_replay.csv.gz")
    b = pd.read_csv(ROOT / "results" / "c2" / "batches.csv.gz")
    res = {"label": "Measured conditions in the valid C1 GPU traces (fixed gamma=2, FCFS). Per-token acceptance beyond position 2, jitter, and the "
                    "prompt-length distribution of the whole catalog are UNKNOWN from these traces.", "per_run": {}}
    pooled_alpha, pooled_n = [], []
    for run in RUNS_TRAIN + ["RQ1_PILOT_4a7e866"]:
        for n in sorted(r[r.run == run].N.unique()) if run in set(r.run) else []:
            g = r[(r.run == run) & (r.N == n)].copy()
            bb = b[(b.run == run) & (b.N == n)]
            w0, w1 = float(bb.experiment_warmup_end_ms.iloc[0]), float(bb.experiment_measurement_end_ms.iloc[0])
            gw = g[(g.client_receive_ms >= w0) & (g.client_receive_ms <= w1)].copy()
            rec = {"rounds_in_window": int(len(gw)), "gamma_values": {int(k): int(v) for k, v in g.gamma.value_counts().items()}}
            g2 = gw[gw.gamma == 2]
            # ---- acceptance
            rows = []
            for cid, x in g2.groupby("client_id"):
                a, nobs = censored_alpha(x.accepted_length.to_numpy(), 2)
                rows.append((cid, a, nobs))
            al = np.array([a for _, a, _ in rows])
            nobs = np.array([k for _, _, k in rows])
            se2 = al * (1 - al) / np.maximum(nobs, 1)
            between_var = max(0.0, float(np.var(al, ddof=1) - se2.mean())) if len(al) > 1 else float("nan")
            rec["alpha_per_client"] = {"n_clients": len(al), "mean": float(np.nanmean(al)), "sd_observed": float(np.nanstd(al, ddof=1)) if len(al) > 1 else None,
                                       "sd_sampling_expected": float(np.sqrt(se2.mean())), "sd_between_client_true_est": float(np.sqrt(between_var)) if between_var == between_var else None,
                                       "min": float(np.nanmin(al)), "max": float(np.nanmax(al)), **q(al)}
            # per-prompt (sample) acceptance within the run
            ps = []
            for (cid, sid), x in g2.groupby(["client_id", "sample_id"]):
                a, k = censored_alpha(x.accepted_length.to_numpy(), 2)
                if k >= 10:
                    ps.append((a, k))
            if ps:
                pa = np.array([a for a, _ in ps])
                pk = np.array([k for _, k in ps])
                pv = max(0.0, float(np.var(pa, ddof=1) - (pa * (1 - pa) / pk).mean()))
                rec["alpha_per_prompt"] = {"n_prompts(>=10 obs)": len(pa), "mean": float(pa.mean()), "sd_observed": float(pa.std(ddof=1)), "sd_between_prompt_true_est": float(np.sqrt(pv)), **q(pa)}
            # positional check (iid-geometric consistency): alpha1 = P(acc>=1); alpha2 = P(acc=2 | acc>=1)
            a1 = float((g2.accepted_length >= 1).mean())
            a2 = float((g2.accepted_length == 2).sum() / max(1, (g2.accepted_length >= 1).sum()))
            rec["positional_alpha"] = {"alpha1": a1, "alpha2_given_1": a2}
            rec["useful_tokens_per_round"] = {"mean": float(g2.committed_token_count.mean()), **q(g2.committed_token_count)}
            pooled_alpha += list(al[~np.isnan(al)])
            pooled_n += list(nobs[~np.isnan(al)])
            # ---- rates and heterogeneity of delivered rate
            dur = (w1 - w0) / 1000.0
            rates = gw.groupby("client_id").committed_token_count.sum() / dur
            rec["rate"] = {"x_min": float(rates.min()), "x_mean": float(rates.mean()), "x_max": float(rates.max()), "x_min_over_mean": float(rates.min() / rates.mean()),
                           "jain": float(rates.sum() ** 2 / (len(rates) * (rates ** 2).sum())), "cv": float(rates.std(ddof=0) / rates.mean())}
            # ---- hardware/config heterogeneity
            rec["config_heterogeneity"] = {"rtt_ms_distinct": int(g.rtt_ms.nunique()), "rtt_ms": float(g.rtt_ms.mean()), "draft_latency_ms_distinct": int(g.draft_latency_ms.nunique()),
                                           "draft_latency_ms": float(g.draft_latency_ms.mean()), "uplink_rate_mbps_range": [float(g.uplink_rate_mbps.min()), float(g.uplink_rate_mbps.max())]}
            # ---- batches / service
            bw = bb[(bb.verify_finish_ms >= w0) & (bb.batch_start_ms <= w1)]
            B = bw.verification_batch_size
            rec["batches"] = {"n": int(len(bw)), "size_hist": {int(k): int(v) for k, v in B.value_counts().sort_index().items()}, "wall_ms": {"mean": float(bw.verification_batch_wall_ms.mean()), **q(bw.verification_batch_wall_ms)},
                              "processed_tokens": {"max": int(bw.verification_batch_processed_tokens.max()), **q(bw.verification_batch_processed_tokens)},
                              "ctx_mean_per_request": {"mean": float((bw.verification_batch_context_tokens_sum / B).mean()), **q(bw.verification_batch_context_tokens_sum / B)},
                              "ctx_max": {"max": int(bw.verification_batch_context_tokens_max.max()), **q(bw.verification_batch_context_tokens_max)}}
            # ---- arrival structure: eligible work at each batch start (members + requests that had arrived but were left for a later batch)
            gg = g.sort_values("server_arrival_ms")
            arr = gg.server_arrival_ms.to_numpy()
            bst = gg.batch_start_ms.to_numpy()
            elig, spreads = [], []
            starts = bw.groupby("verification_batch_id").batch_start_ms.first()
            fin = bw.groupby("verification_batch_id").verify_finish_ms.first()
            for bid, s in starts.items():
                mem = gg[gg.verification_batch_id == bid]
                left = int(((arr <= s + 1e-9) & (bst > s + 1e-9)).sum())
                elig.append(len(mem) + left)
                spreads.append(float(mem.server_arrival_ms.max() - mem.server_arrival_ms.min()))
            elig = np.array(elig)
            # lock-step measure: spread of the members' NEXT arrivals around the batch finish
            nxt = gg.sort_values(["client_id", "server_arrival_ms"])
            nxt["next_arr"] = nxt.groupby("client_id").server_arrival_ms.shift(-1)
            nxt = nxt.dropna(subset=["next_arr"])
            off = (nxt.next_arr - nxt.verify_finish_ms).to_numpy()
            rec["arrival_structure"] = {"eligible_at_batch_start": {"mean": float(elig.mean()), "frac_gt_16": float((elig > 16).mean()), "frac_gt_8": float((elig > 8).mean()), **q(elig)},
                                        "member_arrival_spread_ms": {"mean": float(np.mean(spreads)), **q(spreads)},
                                        "next_arrival_after_finish_ms": {"mean": float(off.mean()), "sd": float(off.std()), **q(off)}}
            res["per_run"][f"{run}:N{n}"] = rec
    pa, pn = np.array(pooled_alpha), np.array(pooled_n)
    res["pooled_gsm8k_train_gamma2"] = {"n_client_estimates": int(len(pa)), "alpha_hat_mean": float(pa.mean()), "alpha_hat_sd": float(pa.std(ddof=1)), "min": float(pa.min()), "max": float(pa.max()), **q(pa)}
    # prompt lengths recoverable from single-member batches (pilot, GSM8K-test): ctx - tokens already generated in the sample
    pl = []
    for n in (1, 2):
        g = r[(r.run == "RQ1_PILOT_4a7e866") & (r.N == n)].sort_values(["client_id", "sample_id", "draft_start_ms"])
        bb = b[(b.run == "RQ1_PILOT_4a7e866") & (b.N == n)].set_index("verification_batch_id")
        g["gen_before"] = g.groupby(["client_id", "sample_id"]).committed_token_count.cumsum() - g.committed_token_count
        one = g[g.verification_batch_size == 1]
        for _, x in one.iterrows():
            ctx = bb.loc[x.verification_batch_id, "verification_batch_context_tokens_sum"]
            pl.append(float(ctx - x.gen_before))
    if pl:
        res["prompt_length_from_single_member_batches_gsm8k_test"] = {"n": len(pl), "mean": float(np.mean(pl)), "sd": float(np.std(pl)), "min": float(np.min(pl)), "max": float(np.max(pl)), **q(pl)}
    # generated tokens per sample, rounds per sample
    gg = r[(r.run == "RQ1_CAP16_5a37f03") & (r.N == 24)]
    per = gg.groupby(["client_id", "sample_id"]).agg(tokens=("committed_token_count", "sum"), rounds=("round_id", "size"))
    res["per_sample_generation_gsm8k_train"] = {"tokens": q(per.tokens), "rounds": q(per.rounds)}
    res["unknown"] = [
        "per-token acceptance at positions 3-4 (traces hold gamma = 2): the gamma=4 acceptance used by any gamma=4 arm is an EXTRAPOLATION of a geometric model",
        "whether acceptance is i.i.d. across positions beyond the 2-position check above",
        "prompt-length distribution of the full GSM8K-train catalog (only hashes are committed; per-request context is observable only through batch aggregates ctx_sum/B and ctx_max)",
        "arrival jitter and device heterogeneity of any real edge deployment (the C1 configs use identical rtt/draft latency for all clients)",
        "GPU memory at max_batch_size = 8 (only cap-16 peaks were measured: see memory evidence)",
    ]
    res["memory_evidence"] = {"cap16_probe_N28_N32_peak_allocated_GiB": [55.98, 55.72], "device_GiB": 63.98, "reserved_peak_GiB": [63.40, 63.45], "after_engine_init_allocated_GiB": 42.8,
                              "headroom_GiB_approx": 0.5, "processed_tokens_max_seen_in_traces": int(b[b.serving.isin(['cap16', 'cap16_probe'])].verification_batch_processed_tokens.max()),
                              "oom_event": "cap-32, N=24: 4.08 GiB allocation in the sampler logprob ranking (batch of 24, ~7,200 positions implied)", "source": "results/AUCSPEC_RQ1_FINAL_SATURATION_PROBE.md, results/AUCSPEC_RQ1_SATURATION_EXPLORATION.md"}
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["per_run"].items():
        if "alpha_per_client" in v:
            a = v["alpha_per_client"]
            print(k, "alpha mean %.3f sd_obs %.3f sd_samp %.3f sd_true %s min %.2f max %.2f" % (a["mean"], a["sd_observed"] or 0, a["sd_sampling_expected"], a["sd_between_client_true_est"], a["min"], a["max"]),
                  "| xmin/mean %.3f jain %.4f" % (v["rate"]["x_min_over_mean"], v["rate"]["jain"]), "| elig>16 %.3f mean %.1f" % (v["arrival_structure"]["eligible_at_batch_start"]["frac_gt_16"], v["arrival_structure"]["eligible_at_batch_start"]["mean"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "c3_1" / "trace_conditions.json"))
