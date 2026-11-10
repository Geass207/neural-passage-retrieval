"""Pairwise distillation trainer for a SentenceTransformer student."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.types import Passage, Query
from src.retrieval.pairdistill.data import PairwiseExample
from src.retrieval.pairdistill.student_retriever import format_passage


@dataclass
class PairDistillTrainingConfig:
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    teacher_temperature: float = 5.0
    student_temperature: float = 1.0
    hard_labels: bool = False
    normalize_embeddings: bool = True
    warmup_ratio: float = 0.0
    seed: int = 42


class _PairwiseDataset(Dataset):
    def __init__(
        self,
        examples: List[PairwiseExample],
        queries: Dict[str, Query],
        corpus: Dict[str, Passage],
    ) -> None:
        self.rows = [
            ex
            for ex in examples
            if ex.qid in queries and ex.pos_pid in corpus and ex.neg_pid in corpus
        ]
        self.queries = queries
        self.corpus = corpus

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        ex = self.rows[idx]
        return {
            "query": self.queries[ex.qid].text,
            "positive": format_passage(self.corpus[ex.pos_pid]),
            "negative": format_passage(self.corpus[ex.neg_pid]),
            "teacher_gap": float(ex.teacher_gap),
        }


def _collate(batch: List[dict]) -> dict:
    return {
        "queries": [row["query"] for row in batch],
        "positives": [row["positive"] for row in batch],
        "negatives": [row["negative"] for row in batch],
        "teacher_gaps": torch.tensor(
            [row["teacher_gap"] for row in batch],
            dtype=torch.float32,
        ),
    }


class PairwiseDistillationTrainer:
    """Train a compact dense retriever from pairwise teacher preferences."""

    def __init__(
        self,
        student_model_name_or_path: str,
        config: Optional[PairDistillTrainingConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or PairDistillTrainingConfig()
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        self.model = SentenceTransformer(student_model_name_or_path, device=device)
        self.device = self.model.device

    def train(
        self,
        examples: List[PairwiseExample],
        queries: Dict[str, Query],
        corpus: Dict[str, Passage],
        output_dir: str,
    ) -> List[dict]:
        dataset = _PairwiseDataset(examples, queries, corpus)
        if len(dataset) == 0:
            raise ValueError("No usable pairwise examples after filtering.")

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._build_scheduler(optimizer, len(loader))

        history: List[dict] = []
        self.model.train()

        for epoch in range(1, self.config.epochs + 1):
            total_loss = 0.0
            total_examples = 0
            progress = tqdm(loader, desc=f"PairDistill epoch {epoch}")

            for batch in progress:
                loss = self._training_step(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                batch_size = len(batch["queries"])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size
                progress.set_postfix(loss=total_loss / max(1, total_examples))

            epoch_log = {
                "epoch": epoch,
                "loss": total_loss / max(1, total_examples),
                "examples": total_examples,
            }
            history.append(epoch_log)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(str(out))
        (out / "pairdistill_training.json").write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "history": history,
                    "num_examples": len(dataset),
                },
                indent=2,
            )
        )
        return history

    def _training_step(self, batch: dict) -> torch.Tensor:
        query_features = batch_to_device(self.model.tokenize(batch["queries"]), self.device)
        pos_features = batch_to_device(self.model.tokenize(batch["positives"]), self.device)
        neg_features = batch_to_device(self.model.tokenize(batch["negatives"]), self.device)

        query_embeddings = self.model(query_features)["sentence_embedding"]
        pos_embeddings = self.model(pos_features)["sentence_embedding"]
        neg_embeddings = self.model(neg_features)["sentence_embedding"]

        if self.config.normalize_embeddings:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            pos_embeddings = F.normalize(pos_embeddings, p=2, dim=1)
            neg_embeddings = F.normalize(neg_embeddings, p=2, dim=1)

        pos_scores = (query_embeddings * pos_embeddings).sum(dim=1)
        neg_scores = (query_embeddings * neg_embeddings).sum(dim=1)
        student_logits = (pos_scores - neg_scores) / self.config.student_temperature

        if self.config.hard_labels:
            targets = torch.ones_like(student_logits)
        else:
            teacher_gaps = batch["teacher_gaps"].to(self.device)
            targets = torch.sigmoid(teacher_gaps / self.config.teacher_temperature)

        return F.binary_cross_entropy_with_logits(student_logits, targets)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, batches_per_epoch: int):
        if self.config.warmup_ratio <= 0:
            return None

        total_steps = max(1, batches_per_epoch * self.config.epochs)
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            remaining = max(1, total_steps - warmup_steps)
            return max(0.0, float(total_steps - step) / float(remaining))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
