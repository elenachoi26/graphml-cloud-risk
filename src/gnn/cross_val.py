"""
The ablation: does graph structure actually carry signal, or is a table enough?

5-fold CV over the 2022–2024 node-year observations, four models, one feature
set. Folds are split over node-year observations rather than over years, because
holding out a whole year would confound "can it generalise structurally" with
"did the graph change between 2023 and 2024".

    python -m src.gnn.cross_val              # all four models
    python -m src.gnn.cross_val --model gcn  # one

Reported result on the real panel (see docs/04-model.md):

    Ridge   ρ = -0.1375   MAE 0.1908   RMSE 0.2179
    MLP     ρ =  0.1750   MAE 0.1896   RMSE 0.2140
    GCN     ρ =  0.8522   MAE 0.0717   RMSE 0.0971

Ridge scoring *below zero* is the informative part: intrinsic 10-K features
alone do not merely under-explain systemic exposure, they point the wrong way.
The firms that describe themselves as most at risk are not the ones the network
actually exposes. That is the argument for the whole approach.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tqdm import tqdm

from .. import config
from .data_loader import RiskDataLoader, NODE_FEAT_COLS
from .graph_builder import HomoGraphBuilder
from .models import BasicGNN, HomoGAT, MLPBaseline, TabularBaseline, evaluate_predictions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over labelled nodes only; unlabelled nodes still pass messages."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


def _evaluate(pred: np.ndarray, true: np.ndarray) -> dict:
    mask = ~np.isnan(true)
    pred_m, true_m = pred[mask], true[mask]
    res = evaluate_predictions(true_m, pred_m)
    # Top-quintile overlap: of the 20% the rule flags as most exposed, how many
    # does the model also flag? This is the number an underwriter cares about.
    k = max(1, int(len(true_m) * 0.2))
    res["top_k_overlap"] = len(
        set(np.argsort(true_m)[-k:]) & set(np.argsort(pred_m)[-k:])
    ) / k
    return res


def _tabular_fold(loader, model_factory, train_obs, val_obs) -> dict:
    weak = loader.load_weak_labels()
    cols = [c for c in NODE_FEAT_COLS if c in weak.columns]
    X = weak[cols].fillna(0.0).values
    y = weak["composite_risk"].values
    model = model_factory(X.shape[1]).fit(X[train_obs.index], y[train_obs.index])
    return _evaluate(model.predict(X[val_obs.index]), y[val_obs.index])


def _graph_fold(graphs, model, train_obs, val_obs, epochs, lr) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pbar = tqdm(range(epochs), desc="  graph", unit="ep", leave=False)
    for _ in pbar:
        total = 0.0
        for yr, data in graphs.items():
            # Validation nodes are masked out of the *loss* but stay in the
            # graph — removing them would change the topology the model sees and
            # make the fold un-representative.
            train_firms = set(train_obs[train_obs["year"] == yr]["canonical_firm_id"])
            y_masked = data.y.clone()
            hide = torch.tensor([c not in train_firms for c in data.canonical_firm_id],
                                dtype=torch.bool).to(DEVICE)
            y_masked[hide] = float("nan")

            model.train()
            optimizer.zero_grad()
            loss = masked_mse_loss(model.forward_data(data), y_masked)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        pbar.set_postfix(loss=f"{total:.4f}")

    preds, trues = [], []
    model.eval()
    for yr, data in graphs.items():
        val_firms = set(val_obs[val_obs["year"] == yr]["canonical_firm_id"])
        sel = [c in val_firms for c in data.canonical_firm_id]
        with torch.no_grad():
            p = model.forward_data(data).cpu().numpy()
        preds.extend(p[sel])
        trues.extend(data.y.cpu().numpy()[sel])
    return _evaluate(np.array(preds), np.array(trues))


def run_kfold(model_type: str = "gcn", k: int = config.CV_FOLDS, epochs: int = 200,
              hidden_dim: int = 64, num_heads: int = 4, lr: float = 1e-3) -> dict:
    loader = RiskDataLoader()
    weak = loader.load_weak_labels()
    obs = weak[["canonical_firm_id", "year", "composite_risk"]].reset_index(drop=True)

    graphs = {}
    if model_type in ("gcn", "gat"):
        builder = HomoGraphBuilder(loader)
        graphs = {yr: builder.build(yr).to(DEVICE) for yr in config.TRAIN_YEARS}
        in_dim = next(iter(graphs.values())).x.shape[1]
        edge_dim = next(iter(graphs.values())).edge_attr.shape[1]

    folds = []
    kf = KFold(n_splits=k, shuffle=True, random_state=config.RANDOM_SEED)
    for i, (tr, va) in enumerate(kf.split(obs), start=1):
        tr_obs, va_obs = obs.iloc[tr], obs.iloc[va]
        print(f"\n  Fold {i}/{k}: train={len(tr_obs)}, val={len(va_obs)}")

        if model_type == "ridge":
            res = _tabular_fold(loader, lambda d: TabularBaseline(), tr_obs, va_obs)
        elif model_type == "mlp":
            res = _tabular_fold(loader, lambda d: MLPBaseline(in_dim=d, hidden_dim=hidden_dim,
                                                              lr=lr, epochs=epochs),
                                tr_obs, va_obs)
        elif model_type == "gcn":
            res = _graph_fold(graphs, BasicGNN(in_dim, hidden_dim).to(DEVICE),
                              tr_obs, va_obs, epochs, lr)
        elif model_type == "gat":
            res = _graph_fold(graphs, HomoGAT(in_dim, hidden_dim, num_heads,
                                              edge_dim=edge_dim).to(DEVICE),
                              tr_obs, va_obs, epochs, lr)
        else:
            raise ValueError(f"unknown model: {model_type}")

        folds.append(res)
        print(f"    ρ={res['spearman']:.4f}  MAE={res['mae']:.4f}  RMSE={res['rmse']:.4f}")

    mean = {m: float(np.mean([f[m] for f in folds])) for m in folds[0]}
    print(f"\n  [{model_type}] mean over {k} folds: "
          f"ρ={mean['spearman']:.4f}  MAE={mean['mae']:.4f}  RMSE={mean['rmse']:.4f}")
    return mean


def run(models=("ridge", "mlp", "gcn"), **kwargs) -> dict:
    results = {m: run_kfold(m, **kwargs) for m in models}

    print("\n" + "=" * 70)
    print(f"{'Model':<10}{'Spearman ρ':>14}{'MAE':>10}{'RMSE':>10}{'Top-20% overlap':>18}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<10}{r['spearman']:>14.4f}{r['mae']:>10.4f}"
              f"{r['rmse']:>10.4f}{r.get('top_k_overlap', float('nan')):>18.4f}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["ridge", "mlp", "gcn", "gat", "all"], default="all")
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    chosen = ("ridge", "mlp", "gcn") if args.model == "all" else (args.model,)
    run(chosen, k=args.folds, epochs=args.epochs, hidden_dim=args.hidden, lr=args.lr)
