"""Mechanism analysis: which language encodes geographic location.

Three sub-analyses, following the dossier in research/mechanism/:

1. Stylometric vs lexical decomposition.
   - Build a ~50-dim hand-crafted stylometric feature vector per description:
     readability indices (textstat), POS distributions and Heylighen-Dewaele
     formality index (NLTK), function-word frequencies (Burrows-Delta basis),
     surface counts (sentence length, type-token ratio, punctuation, etc.).
   - Train a multinomial-LR zip-code classifier on stylometric features only.
   - Compare its accuracy to the full-embedding classifier in causal_inference.
   - The gap quantifies how much of the geographic encoding is sociolinguistic
     vs lexical-semantic.

2. Vocabulary mutual information against zip code.
   - For every word in the corpus, compute IG(word_present; zip_bin).
   - Rank top-100, partition into categories: place names, architectural
     terms, amenity-evocative, aesthetic, formality markers, generic/other.
   - The dossier coding rubric (Hovy 2018) is applied via a regex/lexicon
     prefilter; final coding requires a manual pass we leave for the writeup.

3. (Reserved) Integrated-gradients saliency on the location classifier.
   - Captum is not currently installed; smoke-tests run with sklearn LR
     coefficients as a first-pass saliency proxy. To extend, install captum
     and add LayerIntegratedGradients on a HuggingFace zip-classifier head.

Refs (full dossier in research/mechanism/research_notes.md):
  Hovy & Yang 2021 NAACL — central anchor for stylometric decomposition
  Burrows 2002 LLC — Delta stylometry; function-word MFW basis
  Heylighen & Dewaele 2002 Foundations of Science — formality index F
  Yang & Pedersen 1997 ICML — IG/MI feature selection
  Eisenstein, O'Connor, Smith, Xing 2010 EMNLP — geographic lexical variation
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import textstat
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_RAW_DIR = Path(__file__).resolve().parents[1] / "raw" / "descriptions"
DATA_PROC_DIR = Path(__file__).resolve().parents[1] / "processed"

# Heylighen-Dewaele Penn-Treebank POS classes (NLTK upenn_tagset).
HD_NOUN = {"NN", "NNS", "NNP", "NNPS"}
HD_ADJ = {"JJ", "JJR", "JJS"}
HD_PREP = {"IN"}                         # IN covers prepositions and subord. conj.
HD_ART = {"DT"}                          # determiners stand in for articles in PTB
HD_PRONOUN = {"PRP", "PRP$", "WP", "WP$"}
HD_VERB = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"}
HD_ADV = {"RB", "RBR", "RBS", "WRB"}
HD_INTJ = {"UH"}

PUNCT_CHARS = ",;:!?.-—"


def heylighen_dewaele_F(pos_tags: list[str]) -> float:
    """F = (noun% + adj% + prep% + art% − pronoun% − verb% − adv% − interj% + 100) / 2.

    Higher F => more formal/contextual; lower F => more deictic/conversational.
    Pure tags are token tags (no fractions), so percentages are integer counts /
    total alphanumeric tokens.
    """
    if not pos_tags:
        return float("nan")
    n = len(pos_tags)
    pct = lambda S: 100 * sum(1 for t in pos_tags if t in S) / n
    plus = pct(HD_NOUN) + pct(HD_ADJ) + pct(HD_PREP) + pct(HD_ART)
    minus = pct(HD_PRONOUN) + pct(HD_VERB) + pct(HD_ADV) + pct(HD_INTJ)
    return (plus - minus + 100) / 2


def stylometric_features(text: str, function_words: list[str]) -> dict[str, float]:
    """Compute the per-document stylometric feature vector."""
    text = (text or "").strip()
    if not text:
        return {}

    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    word_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    if not word_tokens:
        return {}

    pos_tags = [t for _, t in nltk.pos_tag(word_tokens)]

    n_words = len(word_tokens)
    n_unique = len(set(w.lower() for w in word_tokens))
    n_chars = sum(len(w) for w in word_tokens)
    n_sents = max(1, len(sentences))

    feats: dict[str, float] = {
        # Readability indices (textstat)
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),

        # Surface
        "n_words": n_words,
        "n_sentences": n_sents,
        "mean_sentence_length": n_words / n_sents,
        "mean_word_length": n_chars / n_words,
        "type_token_ratio": n_unique / n_words,
        "log_n_words": math.log(n_words),

        # Punctuation density
        "comma_per_word": text.count(",") / n_words,
        "semicolon_per_word": text.count(";") / n_words,
        "exclaim_per_word": text.count("!") / n_words,
        "question_per_word": text.count("?") / n_words,

        # POS ratios
        "pos_noun_pct": 100 * sum(1 for t in pos_tags if t in HD_NOUN) / n_words,
        "pos_adj_pct": 100 * sum(1 for t in pos_tags if t in HD_ADJ) / n_words,
        "pos_verb_pct": 100 * sum(1 for t in pos_tags if t in HD_VERB) / n_words,
        "pos_adv_pct": 100 * sum(1 for t in pos_tags if t in HD_ADV) / n_words,
        "pos_prep_pct": 100 * sum(1 for t in pos_tags if t in HD_PREP) / n_words,
        "pos_pronoun_pct": 100 * sum(1 for t in pos_tags if t in HD_PRONOUN) / n_words,
        "pos_det_pct": 100 * sum(1 for t in pos_tags if t in HD_ART) / n_words,

        # Heylighen-Dewaele formality
        "formality_F": heylighen_dewaele_F(pos_tags),
    }

    # Function-word frequency vector (Burrows-Delta-style)
    lower_tokens = [w.lower() for w in word_tokens]
    fw_counter = Counter(t for t in lower_tokens if t in function_words)
    for fw in function_words:
        feats[f"fw_{fw}"] = fw_counter.get(fw, 0) / n_words

    return feats


def build_stylometric_matrix(
    descriptions: list[str], function_words: list[str]
) -> tuple[np.ndarray, list[str]]:
    rows = []
    for i, text in enumerate(descriptions):
        f = stylometric_features(text, function_words)
        if not f:
            f = {}
        rows.append(f)
        if (i + 1) % 100 == 0:
            print(f"    stylometric: {i+1}/{len(descriptions)}")
    df = pd.DataFrame(rows).fillna(0.0)
    feature_names = list(df.columns)
    return df.values.astype(float), feature_names


def load_full_embeddings(city: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (E, zip_labels) for a city's text embedding subset."""
    parquet = DATA_PROC_DIR / f"{city}_embeddings.parquet"
    if not parquet.exists():
        return None
    df = pd.read_parquet(parquet)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols or "zip" not in df.columns:
        return None
    E = df[emb_cols].values.astype(float)
    zips = df["zip"].fillna(0).astype(float).astype(int).astype(str).values
    return E, zips


def classifier_accuracy(X: np.ndarray, y: np.ndarray, k_folds: int = 5) -> dict:
    """Cross-validated multinomial logistic-regression accuracy."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(np.unique(y_enc))
    if n_classes < 2:
        return {"accuracy": float("nan"), "random_baseline": float("nan"),
                "ratio": float("nan"), "n_classes": int(n_classes), "n_obs": int(len(y))}

    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegressionCV(
        Cs=10, max_iter=5000, cv=k_folds, multi_class="auto", n_jobs=-1
    )
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    # Some folds may have classes absent; restrict to classes with >= k_folds members
    counts = pd.Series(y_enc).value_counts()
    keep = counts[counts >= k_folds].index
    mask = np.isin(y_enc, keep)
    if mask.sum() < 50 or len(keep) < 2:
        return {"accuracy": float("nan"), "random_baseline": float("nan"),
                "ratio": float("nan"), "n_classes": int(len(keep)), "n_obs": int(mask.sum())}

    Xs_m = Xs[mask]
    y_m = y_enc[mask]
    scores = cross_val_score(clf, Xs_m, y_m, cv=skf, scoring="accuracy", n_jobs=-1)

    counts_m = pd.Series(y_m).value_counts(normalize=True)
    random_baseline = float((counts_m ** 2).sum())  # majority-class proxy
    uniform_baseline = 1.0 / len(keep)
    return {
        "accuracy": float(scores.mean()),
        "accuracy_sd": float(scores.std()),
        "random_baseline_uniform": uniform_baseline,
        "random_baseline_majority": random_baseline,
        "ratio_uniform": float(scores.mean() / uniform_baseline),
        "n_classes": int(len(keep)),
        "n_obs": int(mask.sum()),
    }


def vocab_mutual_information(
    descriptions: list[str], zips: np.ndarray,
    min_df: int = 5, max_features: int = 10000, top_k: int = 100,
) -> pd.DataFrame:
    le = LabelEncoder()
    y = le.fit_transform(zips)
    cv = CountVectorizer(
        binary=True, lowercase=True, min_df=min_df,
        max_features=max_features, token_pattern=r"\b[a-z]{3,}\b",
    )
    X = cv.fit_transform(descriptions)
    print(f"    vocab MI: vocab size {X.shape[1]}, n_zips {len(np.unique(y))}")
    mi = mutual_info_classif(X, y, discrete_features=True, random_state=42)
    vocab = cv.get_feature_names_out()
    df = (
        pd.DataFrame({"word": vocab, "mi": mi})
        .sort_values("mi", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return df


def categorize_top_words(words: list[str]) -> pd.Series:
    """First-pass automatic categorization. Final coding is manual.

    Categories: place_name, architectural, amenity, aesthetic, formality,
    generic/other.
    """
    PLACE_PATTERNS = re.compile(
        r"^(north|south|east|west|noe|mission|sunset|richmond|marina|nob|"
        r"pacific|presidio|bayview|excelsior|castro|haight|telegraph|"
        r"ingleside|lakeside|outer|inner|downtown|uptown|bernal|potrero|"
        r"glen|st|fremont|brooklyn|bronx|queens|manhattan|bay|midtown|"
        r"upper|lower|williamsburg|astoria|harlem|chelsea|tribeca|soho|"
        r"dumbo|allston|brighton|cambridge|brookline|somerville|jamaica)$",
        re.IGNORECASE,
    )
    ARCH_TERMS = {
        "victorian", "edwardian", "prewar", "postwar", "craftsman", "tudor",
        "colonial", "modern", "loft", "brownstone", "townhouse", "condo",
        "cottage", "mansion", "duplex", "triplex", "stories", "story",
        "shingle", "stucco", "brick", "wood", "frame", "siding", "facade",
    }
    AMEN_TERMS = {
        "park", "subway", "transit", "bart", "muni", "school", "freeway",
        "highway", "shopping", "dining", "restaurants", "cafes", "boutique",
        "steps", "blocks", "near", "close", "minutes", "walk", "walking",
        "trail", "trails", "garden", "yard",
    }
    AESTHETIC = {
        "stunning", "charming", "beautiful", "elegant", "luxurious", "lovely",
        "spectacular", "gorgeous", "exquisite", "rare", "unique", "premier",
        "prestigious", "private", "secluded", "tranquil", "serene", "iconic",
        "renovated", "updated", "remodeled", "pristine", "turnkey",
    }
    FORMAL = {
        "offering", "presents", "boasts", "features", "appointed", "appointments",
        "discerning", "connoisseur", "appreciation", "incomparable",
    }

    def _cat(w: str) -> str:
        if PLACE_PATTERNS.match(w):
            return "place_name"
        if w in ARCH_TERMS:
            return "architectural"
        if w in AMEN_TERMS:
            return "amenity"
        if w in AESTHETIC:
            return "aesthetic"
        if w in FORMAL:
            return "formality"
        return "generic_other"

    return pd.Series([_cat(w) for w in words], name="category")


def build_function_word_list(top_n: int = 200) -> list[str]:
    sw = set(stopwords.words("english"))
    return sorted(sw)[: min(top_n, len(sw))]


def run_mechanism_analysis(city: str, top_k: int = 100) -> dict:
    print(f"\n=== Mechanism analysis: {city} ===")
    desc_path = DATA_RAW_DIR / f"{city}_descriptions.csv"
    if not desc_path.exists():
        return {"city": city, "error": f"missing {desc_path}"}
    df = pd.read_csv(desc_path)
    df = df[df["description"].str.len().fillna(0) > 50].reset_index(drop=True)
    descriptions = df["description"].tolist()
    zips = df["zip"].fillna(0).astype(float).astype(int).astype(str).values
    print(f"  N={len(descriptions):,} descriptions, {len(np.unique(zips))} unique zip codes")

    # 1. Stylometric features
    print("\n  [1] Building stylometric feature matrix...")
    fw_list = build_function_word_list(top_n=200)
    X_style, feat_names = build_stylometric_matrix(descriptions, fw_list)
    print(f"    -> {X_style.shape[1]} stylometric features")

    print("\n  [2] Stylometric-only zip classifier (5-fold CV LR):")
    style_acc = classifier_accuracy(X_style, zips)
    print(f"    accuracy = {style_acc['accuracy']:.3f} ± {style_acc['accuracy_sd']:.3f}, "
          f"vs uniform random {style_acc['random_baseline_uniform']:.3f} "
          f"-> {style_acc['ratio_uniform']:.1f}x random "
          f"(N={style_acc['n_obs']}, K={style_acc['n_classes']})")

    # Full embedding classifier baseline (matches §5.1 of paper)
    print("\n  [3] Full-embedding zip classifier (baseline for comparison):")
    full_emb = load_full_embeddings(city)
    full_acc = None
    if full_emb is not None:
        E, _ = full_emb
        # Align rows: emb parquet preserves original order from descriptions CSV
        n = min(len(descriptions), E.shape[0])
        full_acc = classifier_accuracy(E[:n], zips[:n])
        print(f"    accuracy = {full_acc['accuracy']:.3f} ± {full_acc['accuracy_sd']:.3f} "
              f"-> {full_acc['ratio_uniform']:.1f}x random "
              f"(N={full_acc['n_obs']}, K={full_acc['n_classes']})")
    else:
        print("    full embedding parquet not found")

    # 2. Vocabulary MI ranking
    print("\n  [4] Vocabulary mutual information vs zip code:")
    mi_df = vocab_mutual_information(descriptions, zips, top_k=top_k)
    mi_df["category"] = categorize_top_words(mi_df["word"].tolist())
    cat_counts = mi_df["category"].value_counts().to_dict()
    print(f"    top-{top_k} MI words by category:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"      {cat:18s}  {n:3d}")

    return {
        "city": city,
        "n_descriptions": int(len(descriptions)),
        "n_zips": int(len(np.unique(zips))),
        "n_stylometric_features": int(X_style.shape[1]),
        "stylometric_classifier": style_acc,
        "full_embedding_classifier": full_acc,
        "top_words_by_category": cat_counts,
        "top_words_table": mi_df.to_dict(orient="records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["boston", "nyc", "sf"], required=True)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = run_mechanism_analysis(args.city, top_k=args.top_k)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
