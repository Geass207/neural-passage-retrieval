"""
Reranker Mistral-7B LoRA — second étage de reranking pour la recherche d'information.

Position dans le pipeline : BM25/dense top-100 → ce reranker → top-k reclassé

À l'inférence :
  - Chaque paire (requête, passage) est soumise avec un prompt oui/non
  - Score = logit("yes") - logit("no") sur le premier token généré
  - Les candidats sont reclassés par score décroissant

À l'entraînement (voir scripts/train_mistral_reranker.py) :
  - QLoRA (quantification 4 bits + adaptateurs LoRA) sur Mistral-7B-Instruct-v0.2
  - Perte BCE pointwise sur le token yes/no
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from src.data.types import Query, Results


PROMPT_TEMPLATE = (
    "Query: {query}\n"
    "Passage: {passage}\n"
    "Is this passage relevant to the query? Answer yes or no:"
)


class MistralLoRAReranker:
    """
    Reranker basé sur Mistral-7B-Instruct fine-tuné avec LoRA.

    Peut charger :
      - Un checkpoint d'adaptateur fine-tuné (adapter_path renseigné)
      - Le modèle de base seul pour une comparaison zero-shot

    Utilisation :
        reranker = MistralLoRAReranker(
            adapter_path="results/mistral/training/mistral_lora_r8/best_checkpoint"
        )
        reranker.load()
        results = reranker.rerank(bm25_results, corpus, queries, top_k=10)
    """

    YES_TOKENS = ["yes", "Yes", "YES"]
    NO_TOKENS  = ["no",  "No",  "NO"]

    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        max_length: int = 512,
    ) -> None:
        self.base_model   = base_model
        self.adapter_path = adapter_path
        self.max_length   = max_length
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # La quantification 4 bits requiert CUDA
        self.load_in_4bit = load_in_4bit and (self.device == "cuda")

        self.model     = None
        self.tokenizer = None
        self._yes_ids: List[int] = []
        self._no_ids:  List[int] = []

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Charge le modèle et le tokenizer. À appeler une seule fois avant rerank()."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        print(f"Chargement du tokenizer depuis {self.base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        print(f"Chargement du modèle depuis {self.base_model}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not self.load_in_4bit else None,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.adapter_path and Path(self.adapter_path).exists():
            from peft import PeftModel
            print(f"Chargement de l'adaptateur LoRA depuis {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()

        # Récupère le device réel du modèle (utile avec device_map="auto")
        self._model_device = next(self.model.parameters()).device

        # Mise en cache des ids de tokens yes/no pour éviter de les recalculer à chaque batch
        self._yes_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in self.YES_TOKENS
            if self.tokenizer.encode(t, add_special_tokens=False)
        ]
        self._no_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in self.NO_TOKENS
            if self.tokenizer.encode(t, add_special_tokens=False)
        ]
        print("Modèle prêt.")

    def rerank(
        self,
        first_stage: Results,
        corpus: Dict,
        queries: Dict[str, Query],
        top_k: int = 10,
        batch_size: int = 8,
    ) -> Results:
        """
        Reclasse les résultats du premier étage (BM25 ou dense) avec Mistral-7B.

        Args:
            first_stage : {qid: [(pid, score), ...]} — top-100 issu du BM25/dense
            corpus      : {pid: Passage}
            queries     : {qid: Query}
            top_k       : nombre de passages à conserver par requête en sortie
            batch_size  : nombre de paires (requête, passage) scorées en parallèle

        Retourne :
            {qid: [(pid, score), ...]} trié par score Mistral décroissant, top_k seulement
        """
        if self.model is None:
            raise RuntimeError("Appelez load() avant rerank().")

        results: Results = {}
        for qid, candidates in tqdm(first_stage.items(), desc="Reranking Mistral"):
            if qid not in queries:
                continue
            query_text = queries[qid].text
            pids       = [pid for pid, _ in candidates]
            scores     = self._score_passages(query_text, pids, corpus, batch_size)
            sorted_pairs = sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)
            results[qid] = sorted_pairs[:top_k]

        return results

    # ------------------------------------------------------------------
    # Méthodes privées
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, passage: str) -> str:
        # Limite à 800 caractères pour rester dans max_length avec le prompt
        return PROMPT_TEMPLATE.format(query=query, passage=passage[:800])

    def _score_passages(
        self,
        query: str,
        pids: List[str],
        corpus: Dict,
        batch_size: int,
    ) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(pids), batch_size):
            batch_pids = pids[i : i + batch_size]
            prompts = [
                self._build_prompt(query, corpus[pid].text)
                for pid in batch_pids
                if pid in corpus
            ]
            scores.extend(self._forward_batch(prompts))
        return scores

    @torch.no_grad()
    def _forward_batch(self, prompts: List[str]) -> List[float]:
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        # Envoie les tenseurs sur le device réel du modèle (gère device_map="auto")
        enc = {k: v.to(self._model_device) for k, v in enc.items()}

        outputs = self.model(**enc)
        # Logits sur la dernière position du token, forme (batch, vocab)
        last_logits = outputs.logits[:, -1, :]

        yes_logit = last_logits[:, self._yes_ids].max(dim=-1).values
        no_logit  = last_logits[:, self._no_ids].max(dim=-1).values

        # Log-odds utilisé comme score de pertinence
        return (yes_logit - no_logit).float().cpu().tolist()
