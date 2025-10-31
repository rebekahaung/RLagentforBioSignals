
import argparse
from src.offline.dataset import build_offline_dataset
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_dir', required=True)
    ap.add_argument('--out', default='data/processed/offline_dataset.json')
    ap.add_argument('--policy', choices=['threshold','random'], default='threshold')
    args = ap.parse_args()
    out = build_offline_dataset(args.features_dir, args.out, policy=args.policy)
    print(f"Offline dataset saved to: {out}")
if __name__ == '__main__':
    main()
