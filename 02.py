
import argparse, os, glob, re, json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_dir', required=True)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_json', default='data/processed/hrv/splits.json')
    args = ap.parse_args()
    files = glob.glob(os.path.join(args.features_dir, 'S*_hrv_windows.csv'))
    import numpy as np
    subs = sorted({int(re.search(r'S(\d+)_', os.path.basename(f)).group(1)) for f in files})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * args.val_frac)))
    val = subs[:n_val]; train = subs[n_val:]
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as fp:
        json.dump({'train': train, 'val': val}, fp, indent=2)
    print(f"Splits saved to {args.out_json}: train={train}, val={val}")
if __name__ == '__main__':
    main()
