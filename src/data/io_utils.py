import json
import csv
from pathlib import Path
from typing import Dict, Iterator

from src.data.types import Passage, Query, Qrels


def iter_jsonl(path: str) -> Iterator[dict]:
    """Lit un fichier .jsonl ligne par ligne et retourne un itérateur de dicts."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterator[dict], path: str) -> None:
    """Écrit un itérable de dicts dans un fichier .jsonl."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_corpus_jsonl(path: str) -> Dict[str, Passage]:
    """Charge le corpus depuis un fichier corpus.jsonl au format BEIR."""
    corpus = {}
    for rec in iter_jsonl(path):
        pid = rec["_id"]
        corpus[pid] = Passage(pid=pid, text=rec.get("text", ""), title=rec.get("title", ""))
    return corpus


def load_queries_jsonl(path: str) -> Dict[str, Query]:
    """Charge les requêtes depuis un fichier queries.jsonl au format BEIR."""
    queries = {}
    for rec in iter_jsonl(path):
        qid = rec["_id"]
        queries[qid] = Query(qid=qid, text=rec.get("text", ""))
    return queries


def load_qrels_tsv(path: str) -> Qrels:
    """
    Charge les qrels depuis un fichier TSV.
    Supporte le format BEIR 3 colonnes (qid, pid, score) et le format TREC
    4 colonnes (qid, 0, pid, score) utilisé par MS MARCO.
    """
    qrels: Qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 4:
                # Format TREC : qid  0  pid  score
                qid, pid, score = row[0], row[2], row[3]
            elif len(row) == 3:
                # Format BEIR : qid  pid  score
                qid, pid, score = row[0], row[1], row[2]
            else:
                continue
            qrels.setdefault(qid, {})[pid] = int(score)
    return qrels


def write_qrels_tsv(qrels: Qrels, path: str) -> None:
    """Écrit les qrels dans un fichier TSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid, pid_scores in qrels.items():
            for pid, score in pid_scores.items():
                writer.writerow([qid, pid, score])
