"""
Train and evaluate the BM25-teacher pairwise distillation student.

Example:
    python scripts/run_pairdistill_bm25.py --epochs 2 --batch-size 16
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Optional, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BM25-teacher PairDistill-lite training and evaluation"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--train-split", default=None, help="Split used to sample teacher pairs")
    parser.add_argument("--eval-split", default=None, help="Split used for final evaluation")
    parser.add_argument("--student-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output-dir", default=None, help="Where to save the trained student")
    parser.add_argument("--run-dir", default=None, help="Where to save evaluation artifacts")
    parser.add_argument("--teacher-index-path", default=None, help="BM25 index path")
    parser.add_argument("--teacher-results-path", default=None, help="Optional BM25 teacher cache")
    parser.add_argument("--rebuild-teacher-index", action="store_true")
    parser.add_argument("--use-all-train-queries", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Evaluate an existing student")
    parser.add_argument("--skip-eval", action="store_true", help="Only train and save the student")

    parser.add_argument("--teacher-top-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-eval-queries", type=int, default=None)

    parser.add_argument("--num-pairs-per-query", type=int, default=4)
    parser.add_argument("--positive-pool", type=int, default=10)
    parser.add_argument("--negative-pool-start", type=int, default=20)
    parser.add_argument("--min-rank-gap", type=int, default=5)
    parser.add_argument("--min-score-gap", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--teacher-temperature", type=float, default=5.0)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--hard-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu")

    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--keep-embeddings", action="store_true")
    return parser.parse_args()


def limit_dict(values: Dict[str, T], max_items: Optional[int]) -> Dict[str, T]:
    if max_items is None:
        return values
    return dict(list(values.items())[:max_items])


def limit_corpus_with_qrels(
    corpus: Dict[str, T],
    max_items: Optional[int],
    qrels_sources: Iterable[Optional[Dict[str, Dict[str, int]]]],
) -> Dict[str, T]:
    """
    Limit the corpus while keeping relevant qrels passages whenever possible.

    A naive first-N truncation often removes every relevant passage from the
    evaluation subset, which makes MRR/Recall/nDCG equal to zero regardless of
    model quality.
    """
    if max_items is None:
        return corpus

    selected: Dict[str, T] = {}
    for qrels in qrels_sources:
        if not qrels:
            continue
        for pid_scores in qrels.values():
            for pid, relevance in pid_scores.items():
                if relevance <= 0 or pid not in corpus or pid in selected:
                    continue
                selected[pid] = corpus[pid]
                if len(selected) >= max_items:
                    return selected

    for pid, passage in corpus.items():
        if pid in selected:
            continue
        selected[pid] = passage
        if len(selected) >= max_items:
            break

    return selected


def maybe_load_qrels(dataset, split: str):
    try:
        return dataset.load_qrels(split)
    except FileNotFoundError:
        return None


def main() -> None:
    args = parse_args()

    import yaml

    from src.data.dataset_factory import get_dataset
    from src.evaluation.metrics import evaluate_run
    from src.evaluation.results_io import load_results, save_results, save_run
    from src.retrieval.bm25.bm25_retriever import BM25Retriever
    from src.retrieval.pairdistill.data import build_pairwise_examples
    from src.retrieval.pairdistill.student_retriever import DenseStudentRetriever
    from src.retrieval.pairdistill.trainer import (
        PairDistillTrainingConfig,
        PairwiseDistillationTrainer,
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"]["data_dir"]
    train_split = args.train_split or "train"
    eval_split = args.eval_split or cfg["dataset"]["split"]
    top_k = args.top_k or cfg["retrieval"]["top_k"]
    teacher_index_path = args.teacher_index_path or cfg["bm25"]["index_path"]
    if args.teacher_index_path is None and args.max_corpus is not None:
        teacher_index_path = (
            f"data/{dataset_name}/bm25_index_qrelsaware_maxcorpus{args.max_corpus}.pkl"
        )
    output_dir = args.output_dir or f"models/pairdistill_bm25_{dataset_name}"
    run_dir = args.run_dir or f"results/pairdistill_bm25_{dataset_name}_{eval_split}"
    cache_suffix = f"{dataset_name}_{train_split}"
    if args.max_corpus is not None:
        cache_suffix += f"_qrelsaware_maxcorpus{args.max_corpus}"
    if args.max_train_queries is not None:
        cache_suffix += f"_maxtrainq{args.max_train_queries}"
    teacher_results_path = (
        args.teacher_results_path
        or f"results/pairdistill_bm25_teacher_{cache_suffix}.jsonl"
    )

    print(f"=== PairDistill-lite BM25 | Dataset: {dataset_name} ===")
    print(f"Student: {args.student_model}")
    print(f"Train split: {train_split} | Eval split: {eval_split}")

    dataset = get_dataset(dataset_name, data_dir=data_dir)
    corpus = dataset.load_corpus()
    train_queries = dataset.load_queries(train_split)
    train_qrels = maybe_load_qrels(dataset, train_split)
    if train_qrels is not None and not args.use_all_train_queries:
        train_queries = {qid: q for qid, q in train_queries.items() if qid in train_qrels}
    train_queries = limit_dict(train_queries, args.max_train_queries)
    print(f"Train queries: {len(train_queries):,}")

    eval_queries = dataset.load_queries(eval_split)
    eval_qrels = dataset.load_qrels(eval_split)
    eval_queries = {qid: q for qid, q in eval_queries.items() if qid in eval_qrels}
    eval_queries = limit_dict(eval_queries, args.max_eval_queries)
    eval_qrels = {qid: rels for qid, rels in eval_qrels.items() if qid in eval_queries}

    limited_train_qrels = None
    if train_qrels is not None:
        limited_train_qrels = {
            qid: rels for qid, rels in train_qrels.items() if qid in train_queries
        }
    corpus = limit_corpus_with_qrels(
        corpus,
        args.max_corpus,
        qrels_sources=[limited_train_qrels, eval_qrels],
    )
    print(f"Corpus loaded: {len(corpus):,} passages")

    if not args.skip_train:
        if Path(teacher_results_path).exists():
            print(f"Loading teacher rankings from {teacher_results_path}")
            teacher_results = load_results(teacher_results_path)
            teacher_results = {
                qid: teacher_results[qid]
                for qid in train_queries
                if qid in teacher_results
            }
        else:
            if args.rebuild_teacher_index and Path(teacher_index_path).exists():
                Path(teacher_index_path).unlink()
                print("Existing BM25 index removed.")

            teacher = BM25Retriever(index_path=teacher_index_path)
            teacher.index(corpus)
            teacher_results = teacher.retrieve(train_queries, top_k=args.teacher_top_k)
            save_results(teacher_results, teacher_results_path)
            print(f"Teacher rankings saved to {teacher_results_path}")

        examples = build_pairwise_examples(
            teacher_results=teacher_results,
            num_pairs_per_query=args.num_pairs_per_query,
            positive_pool=args.positive_pool,
            negative_pool_start=args.negative_pool_start,
            min_rank_gap=args.min_rank_gap,
            min_score_gap=args.min_score_gap,
            seed=args.seed,
        )
        print(f"Pairwise examples: {len(examples):,}")
        if not examples:
            raise RuntimeError("No pairwise examples were generated.")

        train_config = PairDistillTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            teacher_temperature=args.teacher_temperature,
            student_temperature=args.student_temperature,
            hard_labels=args.hard_labels,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
        )
        trainer = PairwiseDistillationTrainer(
            student_model_name_or_path=args.student_model,
            config=train_config,
            device=args.device,
        )
        history = trainer.train(examples, train_queries, corpus, output_dir=output_dir)
        print("Training history:")
        print(json.dumps(history, indent=2))

    if args.skip_eval:
        print(f"Training complete. Student saved in: {output_dir}")
        return

    eval_qrels = {
        qid: {pid: rel for pid, rel in rels.items() if pid in corpus}
        for qid, rels in eval_qrels.items()
    }
    eval_qrels = {qid: rels for qid, rels in eval_qrels.items() if rels}
    eval_queries = {qid: query for qid, query in eval_queries.items() if qid in eval_qrels}
    print(f"Eval queries: {len(eval_queries):,}")

    model_path = output_dir if Path(output_dir).exists() else args.student_model
    retriever = DenseStudentRetriever(
        model_name_or_path=model_path,
        batch_size=args.encode_batch_size,
        query_batch_size=args.query_batch_size,
        device=args.device,
        use_faiss=not args.no_faiss,
        keep_embeddings=args.keep_embeddings,
    )
    retriever.index(corpus)
    results = retriever.retrieve(eval_queries, top_k=top_k)
    metrics = evaluate_run(eval_qrels, results, ks=[10, 50, 100])

    print("\n=== Metrics ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    run_config = {
        "system": "pairdistill_bm25_teacher",
        "dataset": dataset_name,
        "train_split": train_split,
        "eval_split": eval_split,
        "student_model": args.student_model,
        "model_path": model_path,
        "teacher_index_path": teacher_index_path,
        "teacher_results_path": teacher_results_path,
        "teacher_top_k": args.teacher_top_k,
        "top_k": top_k,
        "num_pairs_per_query": args.num_pairs_per_query,
        "positive_pool": args.positive_pool,
        "negative_pool_start": args.negative_pool_start,
        "min_rank_gap": args.min_rank_gap,
        "min_score_gap": args.min_score_gap,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "teacher_temperature": args.teacher_temperature,
        "student_temperature": args.student_temperature,
        "hard_labels": args.hard_labels,
        "n_train_queries": len(train_queries),
        "n_eval_queries": len(eval_queries),
        "n_corpus": len(corpus),
    }
    save_run(
        run_dir=run_dir,
        metrics=metrics,
        results=results,
        qrels=eval_qrels,
        queries=eval_queries,
        corpus=corpus,
        config=run_config,
    )
    print(f"\nRun complete. Results in: {run_dir}/")


if __name__ == "__main__":
    main()
