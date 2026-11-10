"""
Fine-tuning QLoRA de Mistral-7B-Instruct comme reranker pointwise sur MS MARCO.

Utilisation (sur serveur GPU) :
    python scripts/reranker/train.py --lora_rank 8          # run principal
    python scripts/reranker/train.py --lora_rank 4          # ablation rank
    python scripts/reranker/train.py --lora_rank 16         # ablation rank
    python scripts/reranker/train.py --smoke_test           # test local sans GPU

Artefacts produits dans results/mistral/training/mistral_lora_r{rank}/ :
    config.json            — hyperparamètres complets du run
    training_loss.json     — perte par step
    metrics_per_epoch.json — métriques par époque
    best_checkpoint/       — poids LoRA du meilleur checkpoint
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.retrieval.reranker.train_dataset import MSMarcoRerankerDataset


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",        default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--lora_rank",         type=int,   default=8,
                   help="Rang LoRA — paramètre principal de l'ablation")
    p.add_argument("--lora_alpha",        type=int,   default=16)
    p.add_argument("--lora_dropout",      type=float, default=0.05)
    p.add_argument("--lr",                type=float, default=2e-4)
    p.add_argument("--epochs",            type=int,   default=2)
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--grad_accum",        type=int,   default=8,
                   help="Batch effectif = batch_size × grad_accum = 32")
    p.add_argument("--warmup_ratio",      type=float, default=0.05)
    p.add_argument("--max_length",        type=int,   default=512)
    p.add_argument("--max_train_samples", type=int,   default=500_000)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--output_dir",        default=None)
    p.add_argument("--smoke_test",        action="store_true",
                   help="Run rapide avec opt-125m pour vérifier le code en local")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Chargement du modèle
# --------------------------------------------------------------------------- #

def load_model_and_tokenizer(args):
    """
    Charge Mistral-7B avec quantification QLoRA 4 bits et adaptateurs LoRA.
    En smoke_test, utilise facebook/opt-125m à la place pour tester le code sans GPU.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    model_name = "facebook/opt-125m" if args.smoke_test else args.base_model
    print(f"Chargement du modèle de base : {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.smoke_test or not torch.cuda.is_available():
        # CPU — pas de quantification
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)

    # Modules ciblés par LoRA — couverture attention uniquement (voir ablation FFN)
    target_modules = ["q_proj", "v_proj"] if args.smoke_test else ["q_proj", "v_proj", "k_proj", "o_proj"]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Fonction de collation
# --------------------------------------------------------------------------- #

def make_collate_fn(tokenizer, max_length: int):
    """
    Construit les batches (input_ids, attention_mask, labels).

    Le label (yes/no) est ajouté à la fin du prompt.
    La perte causal LM est masquée à -100 partout sauf sur le token label,
    ce qui revient à une perte BCE sur le token yes/no.
    """
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("no",  add_special_tokens=False)[0]

    def collate(batch):
        full_texts = [ex["prompt"] + " " + ex["label"] for ex in batch]
        enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        labels = enc["input_ids"].clone()
        for i, ex in enumerate(batch):
            seq_len = enc["attention_mask"][i].sum().item()
            # Masque tout sauf le dernier token réel (le label yes/no)
            labels[i, : seq_len - 1] = -100
            expected = yes_id if ex["label"] == "yes" else no_id
            labels[i, seq_len - 1] = expected
        enc["labels"] = labels
        return enc

    return collate


# --------------------------------------------------------------------------- #
# Évaluation proxy
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_proxy(model, tokenizer, loader, device, yes_ids, no_ids):
    """
    Calcule le log-odds moyen (yes vs no) sur un sous-ensemble du loader.
    Utilisé comme signal de sélection du meilleur checkpoint entre époques.
    """
    model.eval()
    all_logodds = []
    for batch in tqdm(loader, desc="  Éval proxy", leave=False):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**inputs)
        last_logits = outputs.logits[:, -1, :].float()
        yes_score = last_logits[:, yes_ids].max(dim=-1).values
        no_score  = last_logits[:, no_ids].max(dim=-1).values
        all_logodds.extend((yes_score - no_score).cpu().tolist())
    model.train()
    return float(np.mean(all_logodds))


# --------------------------------------------------------------------------- #
# Boucle d'entraînement
# --------------------------------------------------------------------------- #

def train(args) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name   = "smoke_test" if args.smoke_test else f"mistral_lora_r{args.lora_rank}"
    output_dir = Path(args.output_dir or f"results/mistral/training/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Run : {run_name}  |  Sortie : {output_dir}")

    # --- Données ---
    # smoke_test : seulement 64 exemples streamés, pas de téléchargement complet
    max_samples = 64 if args.smoke_test else args.max_train_samples
    train_ds = MSMarcoRerankerDataset(
        max_samples=max_samples,
        seed=args.seed,
    )

    # --- Modèle ---
    model, tokenizer = load_model_and_tokenizer(args)
    collate_fn = make_collate_fn(tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    yes_ids = torch.tensor(
        [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["yes", "Yes"]],
        device=device,
    )
    no_ids = torch.tensor(
        [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["no", "No"]],
        device=device,
    )

    # --- Optimiseur + scheduler cosine ---
    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- Boucle principale ---
    training_loss_log  = []
    metrics_per_epoch  = []
    best_score         = float("-inf")
    global_step        = 0

    model.train()
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Époque {epoch}")):
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            n_batches  += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    training_loss_log.append({
                        "step": global_step,
                        "loss": epoch_loss / n_batches,
                    })

        avg_loss   = epoch_loss / max(n_batches, 1)
        eval_score = evaluate_proxy(model, tokenizer, train_loader, device, yes_ids, no_ids)

        metrics_per_epoch.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "eval_logodds": eval_score,
        })
        print(f"Époque {epoch} | loss={avg_loss:.4f} | eval_logodds={eval_score:.4f}")

        # Sauvegarde du meilleur checkpoint selon le log-odds de validation
        if eval_score > best_score:
            best_score = eval_score
            ckpt_path  = output_dir / "best_checkpoint"
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            print(f"  → Meilleur checkpoint sauvegardé ({ckpt_path})")

    # --- Sauvegarde des logs ---
    (output_dir / "training_loss.json").write_text(json.dumps(training_loss_log, indent=2))
    (output_dir / "metrics_per_epoch.json").write_text(json.dumps(metrics_per_epoch, indent=2))
    print(f"\nEntraînement terminé. Artefacts dans {output_dir}/")


if __name__ == "__main__":
    args = parse_args()
    train(args)
