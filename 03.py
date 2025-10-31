
import argparse, os, glob, json, re
import numpy as np, pandas as pd
from src.env.stress_env import StressRegEnv

def load_sequences(features_dir, splits_json=None, subset='train', max_subjects=5):
    files = glob.glob(os.path.join(features_dir, 'S*_hrv_windows.csv'))
    if splits_json and os.path.exists(splits_json):
        with open(splits_json,'r') as fp: sj=json.load(fp)
        keep=set(sj.get(subset, []))
        files = [f for f in files if int(re.search(r'S(\d+)_', os.path.basename(f)).group(1)) in keep] or files
    files = sorted(files)[:max_subjects]
    seqs = []
    for f in files:
        df = pd.read_csv(f)
        X = df[['hr_mean','sdnn','rmssd']].values.astype('float32')
        seqs.append(X)
    return seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_dir', required=True)
    ap.add_argument('--splits_json', default='data/processed/hrv/splits.json')
    ap.add_argument('--episodes', type=int, default=2)
    args = ap.parse_args()

    seqs = load_sequences(args.features_dir, args.splits_json, subset='train', max_subjects=5)
    env = StressRegEnv(seqs, seed=0)

    for ep in range(args.episodes):
        obs, info = env.reset()
        total = 0.0; steps = 0
        while True:
            a = np.random.randint(0,3)
            obs, r, term, trunc, info = env.step(a)
            total += r; steps += 1
            if term or trunc or steps > 2000: break
        print(f"Episode {ep+1}: steps={steps} total_reward={total:.3f}")

if __name__ == '__main__':
    main()
