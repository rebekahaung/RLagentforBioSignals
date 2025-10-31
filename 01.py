
import argparse, os
from src.features.synth_hrv import generate_hrv_windows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true', help='Generate synthetic HRV windows')
    ap.add_argument('--subjects', nargs='+', type=int, required=True)
    ap.add_argument('--outdir', default='data/processed/hrv')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for s in args.subjects:
        df = generate_hrv_windows(T=800)
        out_csv = os.path.join(args.outdir, f"S{s}_hrv_windows.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} ({len(df)} rows)")

if __name__ == '__main__':
    main()
