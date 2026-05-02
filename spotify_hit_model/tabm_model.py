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

    def fit(self, X: Any, y: Any) -> "TabMClassifier":
        import numpy as np
        import tabm
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        self.classes_ = [0, 1]

        x_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        device = self._resolve_device(torch)
        self.device_ = str(device)

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
            torch.as_tensor(x_array, dtype=torch.float32),
            torch.as_tensor(y_array, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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

            if self.verbose and self.log_every and epoch % self.log_every == 0:
                mean_loss = epoch_loss / max(batches, 1)
                print(f"TabM epoch {epoch}/{self.epochs}: loss={mean_loss:.5f}")

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
