import sys
sys.path.insert(0, 'scripts')
import run_auc_achievable_region as m
import pandas as pd

rounds_csv = m.run_one('mhard_64', 1.0, 64.0, 2)
metrics = m.measure_operating_point(rounds_csv)
row = {'label': 'mhard_64', 'sweep': 'a', 'm_easy': 1.0, 'm_medium': 1.0, 'm_hard': 64.0, 'seed': 2, 'verify_token_budget': 6, **metrics}
pd.DataFrame([row]).to_csv('results/auc_achievable_region/points_sweepa_gapfill.csv', index=False)
print('done', metrics['Y'], metrics['min_x'])
