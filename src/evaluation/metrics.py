"""
Métriques d'évaluation pour la recherche d'information.
Ce module n'a aucune dépendance aux autres modules du projet.
"""
import math
from typing import Dict, List, Tuple

# Types locaux (pas d'import circulaire possible)
Qrels = Dict[str, Dict[str, int]]
Results = Dict[str, List[Tuple[str, float]]]


def recall_at_k(qrels: Qrels, results: Results, k: int) -> float:
    """
    Calcule le Recall@k moyen sur toutes les requêtes.
    Recall@k = |passages pertinents dans top-k| / |passages pertinents totaux|
    """
    scores = []
    for qid, relevant in qrels.items():
        if qid not in results:
            scores.append(0.0)
            continue
        # Seuls les passages avec score > 0 sont considérés pertinents
        relevant_set = {pid for pid, rel in relevant.items() if rel > 0}
        if not relevant_set:
            continue
        retrieved_top_k = {pid for pid, _ in results[qid][:k]}
        scores.append(len(relevant_set & retrieved_top_k) / len(relevant_set))
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(qrels: Qrels, results: Results, k: int = 10) -> float:
    """
    Calcule le MRR@k (Mean Reciprocal Rank) moyen sur toutes les requêtes.
    MRR@k = moyenne de 1/rang du premier passage pertinent dans le top-k.
    """
    scores = []
    for qid, relevant in qrels.items():
        if qid not in results:
            scores.append(0.0)
            continue
        relevant_set = {pid for pid, rel in relevant.items() if rel > 0}
        if not relevant_set:
            continue
        rr = 0.0
        # Parcours du classement jusqu'à k
        for rank, (pid, _) in enumerate(results[qid][:k], start=1):
            if pid in relevant_set:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(qrels: Qrels, results: Results, k: int = 10) -> float:
    """
    Calcule le nDCG@k (Normalized Discounted Cumulative Gain) moyen.
    Prend en compte les scores de pertinence graduels (pas seulement binaires).
    """
    def dcg(relevances: List[int]) -> float:
        """DCG idéalisé sur une liste de pertinences."""
        return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))

    scores = []
    for qid, relevant in qrels.items():
        if qid not in results:
            scores.append(0.0)
            continue
        if not relevant:
            continue

        # Pertinences récupérées dans l'ordre du classement
        retrieved_rels = [relevant.get(pid, 0) for pid, _ in results[qid][:k]]
        # Pertinences idéales (meilleur classement possible)
        ideal_rels = sorted(relevant.values(), reverse=True)[:k]

        ideal_dcg = dcg(ideal_rels)
        if ideal_dcg == 0:
            continue
        scores.append(dcg(retrieved_rels) / ideal_dcg)

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_run(qrels: Qrels, results: Results, ks: List[int] = None) -> Dict[str, float]:
    """
    Lance l'évaluation complète et retourne un dictionnaire de métriques.
    Retourne MRR@10, Recall@{k}, nDCG@10 pour chaque k dans ks.
    """
    if ks is None:
        ks = [10, 50, 100]

    metrics: Dict[str, float] = {
        "MRR@10": mrr_at_k(qrels, results, k=10),
        "nDCG@10": ndcg_at_k(qrels, results, k=10),
    }
    for k in ks:
        metrics[f"Recall@{k}"] = recall_at_k(qrels, results, k=k)

    return metrics
