import yaml

X_POINTS = [2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0]

with open("configs/capacity_frontier_budget100.yaml") as f:
    base = yaml.safe_load(f)

for x in X_POINTS:
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["capacity_search"]["x_grid_final"] = [x]
    tag = str(x).replace(".", "p")
    out_path = f"configs/capacity_frontier_budget100_x{tag}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(out_path)
