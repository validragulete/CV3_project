from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from data_pipeline import prepare_data
from models import CNNClassifier, ConvAutoencoder


def parse_ids(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def classification_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "accuracy": float(accuracy_score(truth, pred)),
        "confusion_matrix_0_1": confusion_matrix(truth, pred, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def cnn_predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        z = model(x.to(device))
        scores.extend(torch.sigmoid(z).cpu().numpy().tolist())
        labels.extend(y.numpy().astype(int).tolist())
    return np.asarray(labels), np.asarray(scores)


@torch.no_grad()
def ae_predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        x = x.to(device)
        errs = torch.mean((model(x) - x) ** 2, dim=(1, 2, 3))
        scores.extend(errs.cpu().numpy().tolist())
        labels.extend(y.numpy().astype(int).tolist())
    return np.asarray(labels), np.asarray(scores)


def select_val_threshold(labels: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) != 2:
        raise ValueError("AE threshold selection needs BOTH normal and abnormal windows in validation")
    uniq = np.unique(errors)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate(([float(uniq[0]) - 1e-8], mids, [float(uniq[-1]) + 1e-8]))
    best = max(((f1_score(labels, errors > t, zero_division=0), float(t)) for t in candidates),
               key=lambda item: (item[0], -item[1]))
    return best[1], float(best[0])


def train_cnn(bundle, device, epochs: int, lr: float):
    model = CNNClassifier().to(device)
    counts = bundle.summary["train"]
    if counts["anomalous"] == 0 or counts["normal"] == 0:
        raise ValueError("Supervised CNN needs normal AND abnormal windows in train")
    pos_weight = torch.tensor([counts["normal"] / counts["anomalous"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1, best_epoch, best_weights = -1.0, 0, None
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in bundle.loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_y, val_prob = cnn_predict(model, bundle.loaders["val"], device)
        val_f1 = float(f1_score(val_y, val_prob >= 0.5, zero_division=0))
        log.append({"model": "CNN", "epoch": epoch, "train_loss": float(np.mean(losses)), "val_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_epoch, best_weights = val_f1, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    return model, log, {"selected_epoch": best_epoch, "val_f1": best_f1, "threshold": 0.5}


def train_ae(bundle, device, epochs: int, lr: float, batch_size: int):
    indices = torch.where(bundle.datasets["train"].labels == 0)[0].tolist()
    if not indices:
        raise ValueError("No clean/normal windows available for AE train")
    loader = DataLoader(Subset(bundle.datasets["train"], indices), batch_size=batch_size, shuffle=True)
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_state, best_epoch = None, 0
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_y, val_errors = ae_predict(model, bundle.loaders["val"], device)
        clean = val_errors[val_y == 0]
        if not len(clean):
            raise ValueError("Autoencoder needs normal windows in validation")
        val_loss = float(clean.mean())
        log.append({"model": "Autoencoder", "epoch": epoch,
                    "train_loss": float(np.mean(losses)), "val_normal_mse": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val_y, val_err = ae_predict(model, bundle.loaders["val"], device)
    threshold, val_f1 = select_val_threshold(val_y, val_err)
    return model, log, {"selected_epoch": best_epoch, "val_normal_mse": best_loss,
                        "val_f1_at_selected_threshold": val_f1, "threshold": threshold}


def make_plots(metrics: dict, logs: list[dict], out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, model in zip(axes, ["CNN", "Autoencoder"]):
        m = metrics[model]["test"]
        cm = np.array(m["confusion_matrix_0_1"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center")
        ax.set(xlabel="Prediction", ylabel="True label", title=model,
               xticks=[0, 1], yticks=[0, 1])
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for model in ["CNN", "Autoencoder"]:
        recs = [r for r in logs if r["model"] == model]
        ax.plot([r["epoch"] for r in recs], [r["train_loss"] for r in recs], label=model)
    ax.set(xlabel="Epoch", ylabel="Training loss (different loss functions)", title="Training progression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Canonical labeled CSV, NOT unmodified raw Cisco ZIP")
    ap.add_argument("--train", default="0,2")
    ap.add_argument("--val", default="4")
    ap.add_argument("--test", default="9")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--step", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {"train": parse_ids(args.train), "val": parse_ids(args.val), "test": parse_ids(args.test)}
    bundle = prepare_data(args.data, splits, args.window, args.step, args.batch_size)
    print("Split audit (partition groups disjoint; chronological scenario-level embargo is upstream):", json.dumps(bundle.summary, indent=2))
    x, y = next(iter(bundle.loaders["train"]))
    print(f"DataLoader batch: X={tuple(x.shape)}, y={tuple(y.shape)}; device={device}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cnn, log_cnn, cnn_info = train_cnn(bundle, device, args.epochs, args.lr)
    ae, log_ae, ae_info = train_ae(bundle, device, args.epochs, args.lr, args.batch_size)
    test_y, cnn_scores = cnn_predict(cnn, bundle.loaders["test"], device)
    test_y_ae, ae_scores = ae_predict(ae, bundle.loaders["test"], device)
    assert np.array_equal(test_y, test_y_ae), "Same test set is required for fair comparison"
    if len(np.unique(test_y)) < 2:
        print("WARNING: Test set has only one class: F1 is not a meaningful comparison")
    metrics = {
        "CNN": {"selection": cnn_info, "test": classification_metrics(test_y, (cnn_scores >= 0.5).astype(int))},
        "Autoencoder": {"selection": ae_info, "test": classification_metrics(test_y, (ae_scores > ae_info["threshold"]).astype(int))},
        "split_audit": bundle.summary,
        "split_groups": bundle.split_groups,
        "features": bundle.feature_names,
        "notes": "AE threshold determined ONLY on validation; test is held out. Event labels must be audited externally. Scenario-proxy labels are NOT event ground truth. Splits may reuse original scenarios in disjoint time blocks.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history = log_cnn + log_ae
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    make_plots(metrics, history, output)
    torch.save(cnn.state_dict(), output / "cnn.pt")
    torch.save(ae.state_dict(), output / "autoencoder.pt")
    print(json.dumps({k: v["test"] for k, v in metrics.items() if k in ("CNN", "Autoencoder")}, indent=2))
    print(f"Saved metrics, model weights and plots in {output.resolve()}")


if __name__ == "__main__":
    main()
