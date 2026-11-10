"""Tokenisation et prétraitement du texte pour BM25."""
import re
import string
from typing import List


# Mots vides anglais courants à supprimer avant indexation
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "they",
}

# Table de traduction pour supprimer la ponctuation rapidement
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokénise un texte en minuscules, supprime la ponctuation et
    optionnellement les mots vides.
    """
    # Mise en minuscules et suppression de la ponctuation
    text = text.lower().translate(_PUNCT_TABLE)
    # Séparation sur les espaces
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def tokenize_batch(texts: List[str], remove_stopwords: bool = True) -> List[List[str]]:
    """Tokénise une liste de textes (utile pour le corpus entier)."""
    return [tokenize(t, remove_stopwords) for t in texts]
