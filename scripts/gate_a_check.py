import sys
sys.path.insert(0, 'scripts')
import run_capacity_frontier as m
import yaml

with open('configs/capacity_frontier_final.yaml') as f:
    base_cfg = yaml.safe_load(f)

ctx = m.RunContext.create(base_cfg, run_tag='gate_a_smoke')
m._ACTIVE_POLICY = 'capacity_dpp'
cache = m.CandidateCache(ctx.cache_path)
meas = m.run_one(ctx.base_cfg, n=8, x=4.0, seed=1, meas_s=60.0,
                  cfg_hash=ctx.config_hash, cache=cache, raw_dir=ctx.raw_dir, force=False)
print('=== Gate A result ===')
print('min_rate_tps', meas.min_rate_tps)
print('per_client_rates', meas.per_client_rates)
print('max_z_slope', meas.max_z_slope)
print('server_queue_slope', meas.server_queue_slope)
print('mean_gamma', meas.mean_gamma)
print('mean_batch_size', meas.mean_batch_size)
print('mean_fill_ratio', meas.mean_fill_ratio)
print('extra', meas.extra)
