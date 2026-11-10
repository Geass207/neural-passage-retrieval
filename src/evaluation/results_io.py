"""Sauvegarde et chargement des résultats d'évaluation au format JSON."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.types import Passage, Query, Qrels, Results


def save_run(
    run_dir: str,
    metrics: Dict[str, float],
    results: Results,
    qrels: Qrels,
    queries: Dict[str, Query],
    corpus: Dict[str, Passage],
    config: Optional[Dict[str, Any]] = None,
    n_qualitative: int = 10,
) -> None:
    """
    Sauvegarde tous les artefacts d'un run dans run_dir :
    config.json, metrics.json, results.jsonl, per_query_metrics.json,
    qualitative_examples.json
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- config.json ---
    run_config = {
        "date": datetime.now().isoformat(),
        **(config or {}),
    }
    (out / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- metrics.json ---
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- results.jsonl ---
    save_results(results, str(out / "results.jsonl"))

    # --- per_query_metrics.json ---
    from src.evaluation.metrics import mrr_at_k, recall_at_k, ndcg_at_k
    per_query = {}
    for qid in qrels:
        q_qrels = {qid: qrels[qid]}
        q_results = {qid: results.get(qid, [])}
        per_query[qid] = {
            "MRR@10": mrr_at_k(q_qrels, q_results, k=10),
            "Recall@100": recall_at_k(q_qrels, q_results, k=100),
            "nDCG@10": ndcg_at_k(q_qrels, q_results, k=10),
        }
    (out / "per_query_metrics.json").write_text(
        json.dumps(per_query, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- qualitative_examples.json ---
    examples = _build_qualitative(results, qrels, queries, corpus, n=n_qualitative)
    (out / "qualitative_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Artefacts sauvegardés dans {out}/")


def _build_qualitative(
    results: Results,
    qrels: Qrels,
    queries: Dict[str, Query],
    corpus: Dict[str, Passage],
    n: int = 10,
) -> List[Dict]:
    """Construit n exemples qualitatifs (requête + top-5 passages récupérés)."""
    examples = []
    for qid in list(qrels.keys())[:n]:
        if qid not in results or qid not in queries:
            continue
        relevant_pids = {pid for pid, rel in qrels[qid].items() if rel > 0}
        top5 = results[qid][:5]
        examples.append({
            "qid": qid,
            "query": queries[qid].text,
            "relevant_pids": list(relevant_pids),
            "retrieved": [
                {
                    "rank": rank,
                    "pid": pid,
                    "score": score,
                    "text": corpus[pid].text[:300] if pid in corpus else "",
                    "relevant": pid in relevant_pids,
                }
                for rank, (pid, score) in enumerate(top5, start=1)
            ],
        })
    return examples


def load_metrics(run_dir: str) -> Dict[str, float]:
    """Charge les métriques depuis un répertoire de run existant."""
    path = Path(run_dir) / "metrics.json"
    return json.loads(path.read_text())


def save_results(results: Results, path: str) -> None:
    """Sauvegarde un ranking complet au format JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for qid, ranking in results.items():
            row = {
                "qid": qid,
                "ranking": [[pid, float(score)] for pid, score in ranking],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_results(path: str) -> Results:
    """Charge un ranking sauvegarde par save_results."""
    results: Results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            results[row["qid"]] = [
                (str(pid), float(score))
                for pid, score in row.get("ranking", [])
            ]
    return results
