
import argparse, torch
from src.offline.cql import cql_train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/processed/offline_dataset.json')
    ap.add_argument('--out', default='data/processed/cql_q.pt')
    ap.add_argument('--metrics', default='data/processed/cql_metrics.json')
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt, metrics = cql_train(args.dataset, args.out, metrics_path=args.metrics, steps=args.steps, alpha=args.alpha, batch_size=args.batch_size, lr=args.lr, device=device)
    print(f"Model saved to: {ckpt}")
    print(f"Metrics saved to: {args.metrics}")
    print('Last metrics:', {k:(metrics[k][-1] if len(metrics[k])>0 else None) for k in ['loss','bellman','cql']})
if __name__ == '__main__':
    main()
