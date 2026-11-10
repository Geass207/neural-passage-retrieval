"""
Télécharge les données MS MARCO passage ranking et les convertit au format BEIR.
Prépare le corpus, les requêtes train/dev et les qrels train/dev-small.
"""
import urllib.request
from pathlib import Path

from tqdm import tqdm

from src.data.io_utils import write_jsonl

MSMARCO_URLS = {
    "corpus": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
    "queries_dev": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
    "qrels_dev": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv",
    "qrels_train": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
}


def _download(url: str, dest: Path, desc: str = "") -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(unit="B", unit_scale=True, desc=desc or dest.name) as bar:
        def _hook(count, block_size, total):
            if bar.total is None and total > 0:
                bar.total = total
            bar.update(count * block_size - bar.n)
        urllib.request.urlretrieve(url, dest, reporthook=_hook)


def _convert_tsv_corpus_to_jsonl(tsv_path: Path, jsonl_path: Path) -> None:
    """Convertit collection.tsv (pid TAB texte) en corpus.jsonl au format BEIR."""
    def _iter():
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Converting corpus"):
                line = line.rstrip("\n")
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    yield {"_id": parts[0], "title": "", "text": parts[1]}
    write_jsonl(_iter(), str(jsonl_path))


def _convert_tsv_queries_to_jsonl(tsv_paths: list[Path], jsonl_path: Path) -> None:
    """Convertit un ou plusieurs fichiers queries*.tsv en queries.jsonl BEIR."""
    def _iter():
        seen = set()
        for tsv_path in sorted(tsv_paths):
            with open(tsv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    parts = line.split("\t", 1)
                    if len(parts) == 2 and parts[0] not in seen:
                        seen.add(parts[0])
                        yield {"_id": parts[0], "text": parts[1]}
    write_jsonl(_iter(), str(jsonl_path))


def download_msmarco(data_dir: str = "data/", rebuild_queries: bool = False) -> None:
    """
    Télécharge et convertit MS MARCO au format BEIR dans data_dir/msmarco/.
    Idempotent : ignore les fichiers déjà présents sur le disque.
    """
    import tarfile

    out = Path(data_dir) / "msmarco"
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # --- corpus ---
    corpus_jsonl = out / "corpus.jsonl"
    if not corpus_jsonl.exists():
        tar_path = tmp / "collection.tar.gz"
        _download(MSMARCO_URLS["corpus"], tar_path, "Downloading corpus")
        print("Extracting corpus...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp)
        tsv_path = tmp / "collection.tsv"
        _convert_tsv_corpus_to_jsonl(tsv_path, corpus_jsonl)
        tsv_path.unlink(missing_ok=True)

    # --- queries train/dev ---
    queries_jsonl = out / "queries.jsonl"
    if rebuild_queries or not queries_jsonl.exists():
        tar_path = tmp / "queries.tar.gz"
        _download(MSMARCO_URLS["queries_dev"], tar_path, "Downloading queries")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp)
        query_tsvs = list(tmp.glob("queries*.tsv"))
        if not query_tsvs:
            raise FileNotFoundError(f"Aucun fichier queries*.tsv trouvé dans {tmp}")
        _convert_tsv_queries_to_jsonl(query_tsvs, queries_jsonl)

    # --- qrels dev ---
    qrels_dev = out / "qrels" / "dev.tsv"
    if not qrels_dev.exists():
        raw = tmp / "qrels.dev.small.tsv"
        _download(MSMARCO_URLS["qrels_dev"], raw, "Downloading dev qrels")
        qrels_dev.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(raw, qrels_dev)

    # --- qrels train ---
    qrels_train = out / "qrels" / "train.tsv"
    if not qrels_train.exists():
        raw = tmp / "qrels.train.tsv"
        _download(MSMARCO_URLS["qrels_train"], raw, "Downloading train qrels")
        qrels_train.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(raw, qrels_train)

    print(f"MS MARCO ready at {out}")


if __name__ == "__main__":
    download_msmarco()
