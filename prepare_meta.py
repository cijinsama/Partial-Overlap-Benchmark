"""
POB Dataset Generator (Command-line version)
============================================

This script builds a toy Partial Overlap Benchmark (POB)-like dataset
using CMU Pronouncing Dictionary and word similarity.
"""

import argparse
import random
from collections import defaultdict

import nltk
import pandas as pd
import tqdm
from nltk.corpus import cmudict
from rapidfuzz.distance import Levenshtein
from wordfreq import top_n_list


def download_resources():
    """Download required NLTK resources."""
    nltk.download("cmudict", quiet=True)


def load_common_words(limit: int = 20000):
    """Load common English words with CMU pronunciations."""
    prondict = cmudict.dict()
    words = top_n_list("en", limit)
    words = [w for w in words if w.isalpha() and w in prondict]
    return words, prondict


def build_length_buckets(words, prondict):
    """Group words into buckets by pronunciation length."""
    buckets = defaultdict(list)
    for word in words:
        pron = prondict[word][0]
        buckets[len(pron)].append((word, pron))
    return buckets


def find_closest_matches(words, prondict, buckets):
    """Find the closest pronunciation match for each word."""
    closest_matches = {}
    for word in tqdm.tqdm(words, desc="Finding closest matches"):
        source_pron = prondict[word][0]
        len_src = len(source_pron)
        candidates = (
            buckets[len_src - 1]
            + buckets[len_src]
            + buckets[len_src + 1]
        )

        min_dist = float("inf")
        closest_word = None
        for cand_word, cand_pron in candidates:
            if cand_word == word:
                continue
            dist = Levenshtein.distance(source_pron, cand_pron)
            if dist < min_dist:
                min_dist = dist
                closest_word = cand_word

        closest_matches[word] = (closest_word, min_dist)
    return closest_matches


def get_pron(word, prondict):
    """Return the first pronunciation of a word."""
    return prondict.get(word.lower(), [[]])[0]


def phrase_pron(phrase, prondict):
    """Get phoneme sequence of a phrase."""
    pron = []
    for w in phrase.split():
        p = get_pron(w, prondict)
        if p:
            pron.extend(p)
    return pron


def build_phrase(similar_dict, prondict, max_len: int = 25):
    """Build a phrase with total phoneme length < max_len."""
    keys = list(similar_dict.keys())
    phrase_words = []
    pron_len = 0
    while True:
        w = random.choice(keys)
        p = get_pron(w, prondict)
        if not p:
            continue
        if pron_len + len(p) < max_len:
            phrase_words.append(w)
            pron_len += len(p)
        else:
            break
    return phrase_words


def replace_one_word(phrase_words, similar_dict):
    """Replace one word in phrase with a random different word."""
    candidates = [i for i, w in enumerate(phrase_words) if w in similar_dict]
    if not candidates:
        return phrase_words

    idx = random.choice(candidates)
    new_phrase = phrase_words.copy()

    random_word = random.choice(list(similar_dict.keys()))
    while random_word == new_phrase[idx]:
        random_word = random.choice(list(similar_dict.keys()))

    new_phrase[idx] = random_word
    return new_phrase


def find_first_diff_index(p1, p2):
    """Return index of first differing phoneme between two sequences."""
    min_len = min(len(p1), len(p2))
    for i in range(min_len):
        if p1[i] != p2[i]:
            return i
    return min_len


def generate_dataset(similar_dict, prondict, num_pairs: int = 1000, max_len: int = 25):
    """Generate dataset of phrase pairs with phoneme differences."""
    dataset = []
    for _ in range(num_pairs):
        phrase1_words = build_phrase(similar_dict, prondict, max_len=max_len)
        phrase2_words = replace_one_word(phrase1_words, similar_dict)
        phrase1 = " ".join(phrase1_words)
        phrase2 = " ".join(phrase2_words)
        pron1 = phrase_pron(phrase1, prondict)
        pron2 = phrase_pron(phrase2, prondict)
        diff_idx = find_first_diff_index(pron1, pron2)
        dataset.append(
            {
                "phrase1": phrase1,
                "phrase2": phrase2,
                "pron1_len": len(pron1),
                "pron2_len": len(pron2),
                "first_diff_phoneme_index": diff_idx,
            }
        )
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate Partial Overlap Benchmark dataset.")
    parser.add_argument("--num_perposition", type=int, default=100, help="Number of phrase pairs for each different position to sample as final output.")
    parser.add_argument("--num_pairs", type=int, default=10000, help="Number of phrase pairs to generate as corpus to sample from later. This is supposed to be much larger than num_perposition * max_len.")
    parser.add_argument("--max_len", type=int, default=25, help="Maximum phoneme length per phrase.")
    parser.add_argument("--output", type=str, default="meta_text.csv", help="Output CSV file path.")
    args = parser.parse_args()

    # Prepare resources
    download_resources()
    words, prondict = load_common_words()
    buckets = build_length_buckets(words, prondict)
    closest_matches = find_closest_matches(words, prondict, buckets)

    # Generate dataset
    data = generate_dataset(closest_matches, prondict, num_pairs=args.num_pairs, max_len=args.max_len)

    # Filter by difference index range
    filtered_data = []
    for i in range(args.max_len):
        sampled = [s for s in data if s["first_diff_phoneme_index"] == i][:args.num_perposition]
        filtered_data.extend(sampled)
        print(f"index {i} sampled {len(sampled)}")

    print(f"Total filtered samples: {len(filtered_data)}")

    # Convert to DataFrame
    df = pd.DataFrame(
        {
            "query_text": [s["phrase1"] for s in filtered_data],
            "anchor_text": [s["phrase2"] for s in filtered_data],
            "match_label": [False for _ in filtered_data],
        }
    )

    # Swap and add positives
    df_swapped = df.rename(
        columns={"query_text": "anchor_text", "anchor_text": "query_text"}
    )
    df_concat = pd.concat([df, df_swapped], ignore_index=True)

    df_same = pd.DataFrame(
        {
            "query_text": df["query_text"].unique(),
            "anchor_text": df["query_text"].unique(),
            "match_label": True,
        }
    )

    df_final = pd.concat([df_concat, df_same], ignore_index=True)
    print(f"Final dataset size: {len(df_final)}")
    df_final.to_csv(args.output, index=False)
    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
