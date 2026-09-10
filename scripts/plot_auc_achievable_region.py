"""Figures 1-5 for AUC_ACHIEVABLE_REGION_REPORT.md."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import glob

OUT = "results/auc_achievable_region"

def load(patt):
    fs = sorted(glob.glob(f"{OUT}/{patt}"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)

aw  = load("points_sweepa_wide10_shard*.csv")    # boost hard_math
amw = load("points_sweepam_wide10_shard*.csv")   # boost medium_cnn_summarize

def mean_by(d, col):
    return d.groupby(col).mean(numeric_only=True).reset_index()

awm  = mean_by(aw, "m_hard")
amwm = mean_by(amw, "m_medium")

# Fig 1: achievable point cloud (x_min vs Y), all probe points both sweeps
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(aw["min_x"], aw["Y"], c="tab:gray", alpha=0.5, label="boost hard_math (per seed)")
ax.scatter(amw["min_x"], amw["Y"], c="tab:blue", alpha=0.5, label="boost medium_cnn (per seed)")
ax.scatter(amwm["min_x"], amwm["Y"], c="tab:red", s=90, zorder=5, label="boost medium_cnn (mean per m)")
for _,r in amwm.iterrows():
    ax.annotate(f"m={int(r['m_medium'])}", (r["min_x"], r["Y"]), fontsize=8, xytext=(4,4), textcoords="offset points")
ax.set_xlabel("achieved system min_x  (tok/s/client)"); ax.set_ylabel("system goodput Y  (tok/s)")
ax.set_title("Fig 1: Achievable (x_min, Y) point cloud (verify_token_budget=10)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_point_cloud.png", dpi=130); plt.close(fig)

# Fig 2: empirical upper envelope Y_hat_star(x)
allpts = pd.concat([aw, amw], ignore_index=True)
xs = sorted(allpts["min_x"].unique())
env = [(x, allpts.loc[allpts["min_x"]>=x, "Y"].max()) for x in xs]
ex, ey = zip(*env)
fig, ax = plt.subplots(figsize=(7,5))
ax.step(ex, ey, where="post", c="tab:red", lw=2, label=r"$\hat Y^\star(x)$")
ax.scatter(allpts["min_x"], allpts["Y"], c="tab:gray", alpha=0.35, s=20)
ax.set_xlabel("x  (min interactivity requirement, tok/s/client)"); ax.set_ylabel(r"$\hat Y^\star(x)$  (tok/s)")
ax.set_title("Fig 2: Empirical upper envelope over all probe points")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig2_upper_envelope.png", dpi=130); plt.close(fig)

# Fig 3: bottleneck-group service response vs multiplier (both sweeps, on their own target group)
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(awm["m_hard"], awm["verifier_token_share_hard_math"], "o-", c="tab:gray", label="hard_math vtok share vs m_hard")
ax.plot(awm["m_hard"], awm["share_hard_math"], "s--", c="tab:gray", label="hard_math service share vs m_hard")
ax.plot(amwm["m_medium"], amwm["verifier_token_share_medium_cnn_summarize"], "o-", c="tab:blue", label="cnn vtok share vs m_medium")
ax.plot(amwm["m_medium"], amwm["share_medium_cnn_summarize"], "s--", c="tab:blue", label="cnn service share vs m_medium")
ax.set_xscale("log", base=2); ax.set_xlabel("class priority multiplier m"); ax.set_ylabel("share of that class")
ax.set_title("Fig 3: Target-group service response to its own multiplier")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/fig3_group_response.png", dpi=130); plt.close(fig)

# Fig 4: mechanism plot for the TRUE-bottleneck sweep (am): min_x, eta, Y vs m_medium
fig, ax1 = plt.subplots(figsize=(7,5))
ax1.plot(amwm["m_medium"], amwm["min_x"], "o-", c="tab:red", label="min_x")
ax1.set_xscale("log", base=2); ax1.set_xlabel("m_medium (boost on true bottleneck class)")
ax1.set_ylabel("min_x (tok/s/client)", color="tab:red"); ax1.tick_params(axis="y", labelcolor="tab:red")
ax2 = ax1.twinx()
ax2.plot(amwm["m_medium"], amwm["Y"], "s-", c="tab:blue", label="Y")
ax2.plot(amwm["m_medium"], amwm["eta"]*50, "^--", c="tab:green", label="eta x50")
ax2.set_ylabel("Y (tok/s)  /  eta x50", color="tab:blue"); ax2.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_title("Fig 4: Mechanism -- min_x, Y, eta vs m_medium (Outcome C: min_x up, Y up, eta flat)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_mechanism.png", dpi=130); plt.close(fig)

# Fig 5: per-class x_min vs m_medium (does the floor actually lift?)
fig, ax = plt.subplots(figsize=(7,5))
for cls, c in [("easy_gsm8k","tab:green"), ("medium_cnn_summarize","tab:red"), ("hard_math","tab:blue")]:
    ax.plot(amwm["m_medium"], amwm[f"x_min_{cls}"], "o-", c=c, label=cls)
ax.set_xscale("log", base=2); ax.set_xlabel("m_medium"); ax.set_ylabel("per-class min x_i (tok/s/client)")
ax.set_title("Fig 5: Per-class interactivity floor vs m_medium")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig5_per_class.png", dpi=130); plt.close(fig)

print("wrote fig1..fig5 png under", OUT)
