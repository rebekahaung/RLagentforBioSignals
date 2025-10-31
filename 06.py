
import argparse, os, glob, json, re, numpy as np, pandas as pd, torch
from src.env.stress_env import StressRegEnv
from src.offline.cql import load_policy

def load_sequences(features_dir, splits_json=None, subset='val', max_subjects=5):
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
    ap.add_argument('--ckpt', default='data/processed/cql_q.pt')
    ap.add_argument('--episodes', type=int, default=5)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    q = load_policy(args.ckpt, device=device); q.to(device)

    seqs = load_sequences(args.features_dir, args.splits_json, subset='val', max_subjects=5)
    env = StressRegEnv(seqs, seed=123)

    def policy(obs):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            qvals = q(x).squeeze(0).cpu().numpy()
        return int(np.argmax(qvals))

    totals = []
    for ep in range(args.episodes):
        obs,_ = env.reset(); total=0.0; steps=0
        while True:
            a = policy(obs)
            obs,r,term,trunc,_ = env.step(a)
            total += r; steps += 1
            if term or trunc or steps>2000: break
        print(f"Episode {ep+1}: steps={steps} total_reward={total:.3f}")
        totals.append(total)
    print("Avg total reward:", float(np.mean(totals)))

if __name__ == '__main__':
    main()
