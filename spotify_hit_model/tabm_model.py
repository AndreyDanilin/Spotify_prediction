from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TabMClassifier:
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 0.002
    weight_decay: float = 0.0003
    random_state: int = 21
    k: int = 32
    n_blocks: int = 3
    d_block: int = 512
    dropout: float = 0.1
    device: str = "auto"
    verbose: bool = False
    log_every: int = 10
    early_stopping: bool = True
    validation_fraction: float = 0.15
    patience: int = 10
    min_delta: float = 1e-4

    def fit(self, X: Any, y: Any) -> "TabMClassifier":
        import copy
        import numpy as np
        import tabm
        import torch
        import torch.nn.functional as F
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        self.classes_ = [0, 1]

        x_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        device = self._resolve_device(torch)
        self.device_ = str(device)
        train_x, valid_x, train_y, valid_y = self._split_train_validation(
            x_array,
            y_array,
            train_test_split,
        )

        model = tabm.TabM.make(
            n_num_features=x_array.shape[1],
            d_out=1,
            k=self.k,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = TensorDataset(
            torch.as_tensor(train_x, dtype=torch.float32),
            torch.as_tensor(train_y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        valid_tensor_x = (
            torch.as_tensor(valid_x, dtype=torch.float32).to(device)
            if valid_x is not None
            else None
        )
        valid_tensor_y = (
            torch.as_tensor(valid_y, dtype=torch.float32).to(device)
            if valid_y is not None
            else None
        )
        best_state = None
        best_validation_loss = float("inf")
        stale_epochs = 0
        self.n_epochs_ = 0
        self.best_validation_loss_ = None

        model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            batches = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = model(x_batch).squeeze(-1)
                target = y_batch.unsqueeze(1).expand_as(logits)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batches += 1

            self.n_epochs_ = epoch
            validation_loss = None
            if valid_tensor_x is not None and valid_tensor_y is not None:
                model.eval()
                with torch.inference_mode():
                    valid_logits = model(valid_tensor_x).squeeze(-1)
                    valid_target = valid_tensor_y.unsqueeze(1).expand_as(valid_logits)
                    validation_loss = float(
                        F.binary_cross_entropy_with_logits(
                            valid_logits,
                            valid_target,
                        ).detach().cpu()
                    )
                model.train()
                if validation_loss < best_validation_loss - self.min_delta:
                    best_validation_loss = validation_loss
                    best_state = copy.deepcopy(model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1

            if self.verbose and self.log_every and epoch % self.log_every == 0:
                mean_loss = epoch_loss / max(batches, 1)
                suffix = (
                    f", val_loss={validation_loss:.5f}"
                    if validation_loss is not None
                    else ""
                )
                print(f"TabM epoch {epoch}/{self.epochs}: loss={mean_loss:.5f}{suffix}")

            if (
                self.early_stopping
                and valid_tensor_x is not None
                and stale_epochs >= self.patience
            ):
                if self.verbose:
                    print(f"TabM early stopping at epoch {epoch}/{self.epochs}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            self.best_validation_loss_ = best_validation_loss

        self.model_ = model.to("cpu").eval()
        self.n_features_in_ = x_array.shape[1]
        return self

    def predict_proba(self, X: Any):
        import numpy as np
        import torch

        if not hasattr(self, "model_"):
            raise RuntimeError("TabMClassifier must be fitted before prediction")

        x_array = np.asarray(X, dtype=np.float32)
        with torch.inference_mode():
            logits = self.model_(torch.as_tensor(x_array, dtype=torch.float32)).squeeze(-1)
            positive = torch.sigmoid(logits).mean(dim=1).cpu().numpy()
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, X: Any):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _resolve_device(self, torch: Any):
        requested = self.device.lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TabM device='cuda' was requested, but CUDA is not available")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("TabM device must be one of: auto, cpu, cuda")
        return torch.device(requested)

    def _split_train_validation(self, x_array: Any, y_array: Any, train_test_split: Any):
        import numpy as np

        if not self.early_stopping or self.validation_fraction <= 0:
            return x_array, None, y_array, None
        if len(y_array) < 20:
            return x_array, None, y_array, None

        unique, counts = np.unique(y_array.astype(int), return_counts=True)
        stratify = y_array if len(unique) == 2 and counts.min() >= 2 else None
        return train_test_split(
            x_array,
            y_array,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=stratify,
        )
