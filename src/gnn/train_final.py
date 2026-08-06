"""
Retrain the selected GCN on the full 2022–2024 weak-labelled graph set.

Cross-validation chose the architecture; this run uses every labelled node,
because the point is not another estimate of generalisation error but the best
model to carry onto the 2025 graph.

    python -m src.gnn.train_final --epochs 100 --hidden 32 --lr 0.005

Writes `outputs/models/final_gcn_model.pt`.
"""

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from .. import config
from .cross_val import masked_mse_loss
from .data_loader import RiskDataLoader, NODE_FEAT_COLS
from .graph_builder import HomoGraphBuilder
from .models import BasicGNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(epochs: int = config.GNN_EPOCHS, hidden_dim: int = config.GNN_HIDDEN,
        lr: float = config.GNN_LR) -> BasicGNN:
    print("=" * 70)
    print("Final GCN Training")
    print("=" * 70)
    print(f"  device      : {DEVICE}")
    print(f"  train years : {config.TRAIN_YEARS}")
    print(f"  epochs={epochs}  hidden={hidden_dim}  lr={lr}")
    print(f"  features    : {NODE_FEAT_COLS}")

    loader = RiskDataLoader()
    builder = HomoGraphBuilder(loader)
    graphs = {yr: builder.build(yr).to(DEVICE) for yr in config.TRAIN_YEARS}

    in_dim = graphs[config.TRAIN_YEARS[0]].x.shape[1]
    for yr, data in graphs.items():
        labelled = int((~torch.isnan(data.y)).sum().item())
        print(f"  {yr}: nodes={data.x.shape[0]}, edges={data.edge_index.shape[1]}, "
              f"labels={labelled}")

    model = BasicGNN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pbar = tqdm(range(epochs), desc="Final GCN", unit="ep", dynamic_ncols=True)
    for _ in pbar:
        total = 0.0
        # Each year is a separate graph, so an epoch is one step per snapshot
        # rather than one step over a merged graph — years must not exchange
        # messages with each other.
        for data in graphs.values():
            model.train()
            optimizer.zero_grad()
            loss = masked_mse_loss(model.forward_data(data), data.y)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        pbar.set_postfix(loss=f"{total:.6f}")

    config.ensure_dirs()
    path = config.MODEL_DIR / "final_gcn_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": "BasicGNN_GCN",
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "train_years": config.TRAIN_YEARS,
        "feature_cols": NODE_FEAT_COLS,
    }, path)
    print(f"\n  → {path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=config.GNN_EPOCHS)
    parser.add_argument("--hidden", type=int, default=config.GNN_HIDDEN)
    parser.add_argument("--lr", type=float, default=config.GNN_LR)
    args = parser.parse_args()
    run(epochs=args.epochs, hidden_dim=args.hidden, lr=args.lr)
