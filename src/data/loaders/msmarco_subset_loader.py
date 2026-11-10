"""
MS MARCO Subset Loader — loads a percentage sample for faster development.
Maintains query → relevant passage relationships.

Usage:
    from src.data.dataset_factory import get_dataset

    # 10% subset
    dataset = get_dataset("msmarco_subset", data_dir="data/", subset_pct=0.1)
    corpus, queries, qrels = dataset.get_split("dev")

    # 5% subset with different seed
    dataset = get_dataset("msmarco_subset", data_dir="data/", subset_pct=0.05, seed=123)
"""
from pathlib import Path
from typing import Dict, Tuple
import random

from src.data.base_dataset import BaseDataset
from src.data.io_utils import load_corpus_jsonl, load_queries_jsonl, load_qrels_tsv
from src.data.types import Passage, Query, Qrels


class MSMarcoSubsetLoader(BaseDataset):
    """
    Load a percentage sample of MS MARCO.
    - Samples X% of queries
    - Keeps only passages relevant to sampled queries

    This is much faster for development/testing while maintaining the same data format.
    """

    def __init__(self, data_dir: str = "data/", subset_pct: float = 0.1, seed: int = 42):
        """
        Args:
            data_dir: Root data directory
            subset_pct: Percentage to sample (0.1 = 10%, 0.05 = 5%, etc.)
            seed: Random seed for reproducibility
        """
        self.root = Path(data_dir) / "msmarco"
        self.subset_pct = subset_pct
        self.seed = seed
        random.seed(seed)

    def _ensure_downloaded(self) -> None:
        """Check if MS MARCO files exist, download if not."""
        corpus = self.root / "corpus.jsonl"
        queries = self.root / "queries.jsonl"
        qrels_dev = self.root / "qrels" / "dev.tsv"
        if not (corpus.exists() and queries.exists() and qrels_dev.exists()):
            print("MS MARCO files not found — downloading...")
            from src.data.loaders.msmarco_downloader import download_msmarco
            download_msmarco(str(self.root.parent))

    def load_corpus(self) -> Dict[str, Passage]:
        """Load FULL corpus (filtering happens in get_split)."""
        self._ensure_downloaded()
        return load_corpus_jsonl(str(self.root / "corpus.jsonl"))

    def load_queries(self, split: str = "dev") -> Dict[str, Query]:
        """Load FULL queries."""
        self._ensure_downloaded()
        return load_queries_jsonl(str(self.root / "queries.jsonl"))

    def load_qrels(self, split: str = "dev") -> Qrels:
        """Load FULL qrels."""
        self._ensure_downloaded()
        path = self.root / "qrels" / f"{split}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Qrels not found: {path}")
        return load_qrels_tsv(str(path))

    def get_split(self, split: str = "dev") -> Tuple[Dict[str, Passage], Dict[str, Query], Qrels]:
        """
        Override to return a sampled subset while maintaining relationships.
        
        Sampling strategy:
        1. Sample X% of queries (from qrels of this split)
        2. Keep ALL relevant passages for those queries
        3. Add random negative passages to reach ~X% of total corpus
        
        This ensures realistic negative sampling for training.
        """
        print(f"[LOADING] MS MARCO {split} split (sampling {self.subset_pct*100:.0f}%)...")
        
        # Load full data
        corpus_full = self.load_corpus()
        queries_full = self.load_queries(split)
        qrels_full = self.load_qrels(split)

        # Sample X% of QRELS queries (from this split only)
        qrel_qids = list(qrels_full.keys())
        n_sample_queries = max(1, int(len(qrel_qids) * self.subset_pct))
        sampled_qids = set(random.sample(qrel_qids, n_sample_queries))

        # Filter qrels to sampled queries
        qrels_subset = {qid: qrels_full[qid] for qid in sampled_qids}

        # Get all relevant passage IDs from the sampled qrels
        relevant_pids = set()
        for qid, pid_scores in qrels_subset.items():
            relevant_pids.update(pid_scores.keys())

        # Calculate target number of passages (~X% of corpus)
        target_passages = max(len(relevant_pids), int(len(corpus_full) * self.subset_pct))
        
        # If we need more passages, add random negatives
        if len(relevant_pids) < target_passages:
            non_relevant_pids = set(corpus_full.keys()) - relevant_pids
            n_negatives = target_passages - len(relevant_pids)
            sampled_negatives = set(random.sample(list(non_relevant_pids), 
                                                   min(n_negatives, len(non_relevant_pids))))
            all_pids = relevant_pids | sampled_negatives
        else:
            all_pids = relevant_pids

        # Filter corpus and queries
        corpus_subset = {pid: corpus_full[pid] for pid in all_pids if pid in corpus_full}
        queries_subset = {qid: queries_full[qid] for qid in sampled_qids if qid in queries_full}

        print(f"  [OK] Qrels sampled:    {len(qrel_qids):,} -> {len(qrels_subset):,}")
        print(f"  [OK] Queries:          {len(queries_full):,} -> {len(queries_subset):,}")
        print(f"  [OK] Passages:         {len(corpus_full):,} -> {len(corpus_subset):,}")
        print(f"       (Relevant: {len(relevant_pids):,}, Negatives: {len(corpus_subset) - len(relevant_pids):,})")

        return corpus_subset, queries_subset, qrels_subset

