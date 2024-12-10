from functools import lru_cache
from itertools import product
from typing import List, Set, Tuple
import numpy as np
from pandas import DataFrame
from preprocessor import Preprocessor
import nltk
from nltk.util import bigrams
from difflib import SequenceMatcher
from nltk.metrics import jaccard_distance

@lru_cache(maxsize=None)
def cached_wup_similarity(syn1, syn2):
    """
    Cache Wu-Palmer similarity computation for synsets.
    """
    return syn1.wup_similarity(syn2)

class FeatureExtractor:
    # POS tags
    NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
    VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    ADJECTIVES = ['JJ', 'JJR', 'JJS']
    ADVERBS = ['RB', 'RBR', 'RBS']
    VERBS_PRESENT = ['VBG', 'VBP', 'VBZ']
    VERBS_PAST = ['VBD', 'VBN']
    VERBS_OTHER = ['VB']

    def __init__(self):
        self.preprocessor = Preprocessor()

    def add_POS_statistics(self, df) -> None:
        """
        Add POS-based features to the DataFrame.
        """
        print("Adding POS based features...")

        pos = self.preprocessor.preprocess_df(df, 'lowercase_tokenize_POS')

        pos_features = {
            's1_n_words': [],
            's2_n_words': [],
            's1_n_verbs_tot': [],
            's2_n_verbs_tot': [],
            's1_n_verbs_pres': [],
            's2_n_verbs_pres': [],
            's1_n_verbs_past': [],
            's2_n_verbs_past': [],
            's1_n_nouns': [],
            's2_n_nouns': [],
            's1_n_adjectives': [],
            's2_n_adjectives': [],
            's1_n_adverbs': [],
            's2_n_adverbs': [],
            'dif_n_words': [],
            'dif_n_verbs_tot': [],
            'dif_n_verbs_pres': [],
            'dif_n_verbs_past': [],
            'dif_n_nouns': [],
            'dif_n_adjectives': [],
            'dif_n_adverbs': [],
            'jaccard_all_words': [],
            'jaccard_verbs': [],
            'jaccard_nouns': [],
            'jaccard_adjectives': [],
            'jaccard_adverbs': [],
        }

        def safe_jaccard(set1, set2):
            """
            Safely calculate Jaccard similarity.
            """
            return 1 - jaccard_distance(set1, set2) if set1 and set2 else 0
    
        for pair in pos:
            s1_tokens = pair[0]
            s2_tokens = pair[1]

            # Word counts
            pos_features['s1_n_words'].append(len(s1_tokens))
            pos_features['s2_n_words'].append(len(s2_tokens))

            # Count specific POS categories
            s1_verbs = [word for word in s1_tokens if word[1] in self.VERBS]
            s2_verbs = [word for word in s2_tokens if word[1] in self.VERBS]

            s1_verbs_pres = [word for word in s1_tokens if word[1] in self.VERBS_PRESENT]
            s2_verbs_pres = [word for word in s2_tokens if word[1] in self.VERBS_PRESENT]

            s1_verbs_past = [word for word in s1_tokens if word[1] in self.VERBS_PAST]
            s2_verbs_past = [word for word in s2_tokens if word[1] in self.VERBS_PAST]

            s1_nouns = [word for word in s1_tokens if word[1] in self.NOUNS]
            s2_nouns = [word for word in s2_tokens if word[1] in self.NOUNS]

            s1_adjectives = [word for word in s1_tokens if word[1] in self.ADJECTIVES]
            s2_adjectives = [word for word in s2_tokens if word[1] in self.ADJECTIVES]

            s1_adverbs = [word for word in s1_tokens if word[1] in self.ADVERBS]
            s2_adverbs = [word for word in s2_tokens if word[1] in self.ADVERBS]

            # Append counts
            pos_features['s1_n_verbs_tot'].append(len(s1_verbs))
            pos_features['s2_n_verbs_tot'].append(len(s2_verbs))

            pos_features['s1_n_verbs_pres'].append(len(s1_verbs_pres))
            pos_features['s2_n_verbs_pres'].append(len(s2_verbs_pres))

            pos_features['s1_n_verbs_past'].append(len(s1_verbs_past))
            pos_features['s2_n_verbs_past'].append(len(s2_verbs_past))

            pos_features['s1_n_nouns'].append(len(s1_nouns))
            pos_features['s2_n_nouns'].append(len(s2_nouns))

            pos_features['s1_n_adjectives'].append(len(s1_adjectives))
            pos_features['s2_n_adjectives'].append(len(s2_adjectives))

            pos_features['s1_n_adverbs'].append(len(s1_adverbs))
            pos_features['s2_n_adverbs'].append(len(s2_adverbs))

            # Differences
            pos_features['dif_n_words'].append(len(s1_tokens) - len(s2_tokens))
            pos_features['dif_n_verbs_tot'].append(len(s1_verbs) - len(s2_verbs))
            pos_features['dif_n_verbs_pres'].append(len(s1_verbs_pres) - len(s2_verbs_pres))
            pos_features['dif_n_verbs_past'].append(len(s1_verbs_past) - len(s2_verbs_past))
            pos_features['dif_n_nouns'].append(len(s1_nouns) - len(s2_nouns))
            pos_features['dif_n_adjectives'].append(len(s1_adjectives) - len(s2_adjectives))
            pos_features['dif_n_adverbs'].append(len(s1_adverbs) - len(s2_adverbs))

            # Jaccard similarities
            pos_features['jaccard_all_words'].append(safe_jaccard(set(s1_tokens), set(s2_tokens)))
            pos_features['jaccard_verbs'].append(safe_jaccard(set(s1_verbs), set(s2_verbs)))
            pos_features['jaccard_nouns'].append(safe_jaccard(set(s1_nouns), set(s2_nouns)))
            pos_features['jaccard_adjectives'].append(safe_jaccard(set(s1_adjectives), set(s2_adjectives)))
            pos_features['jaccard_adverbs'].append(safe_jaccard(set(s1_adverbs), set(s2_adverbs)))

        for key, values in pos_features.items():
            df[key] = values


    def compute_synset_distances(self, synsets1, synsets2):
        """
        Compute synset-based similarity statistics.
        """
        shared_synsets = synsets1.intersection(synsets2)
        total_unique_synsets = len(synsets1 | synsets2)
        shared_ratio = len(shared_synsets) / total_unique_synsets if total_unique_synsets > 0 else 0

        similarities = [
            cached_wup_similarity(syn1, syn2)
            for syn1, syn2 in product(synsets1, synsets2)
            if syn1.pos() == syn2.pos() and cached_wup_similarity(syn1, syn2) is not None
        ]

        avg_similarity = sum(similarities) / len(similarities) if similarities else (1 if not synsets1 and not synsets2 else 0)
        max_similarity = max(similarities) if similarities else avg_similarity

        return {
            "shared_synsets_count": len(shared_synsets),
            "shared_synsets_ratio": shared_ratio,
            "avg_synset_similarity": avg_similarity,
            "max_synset_similarity": max_similarity,
        }


    def add_synset_statistics(self, df: DataFrame) -> None:
        """
        Add synset-based features to the DataFrame.
        """
        print("Adding synset-based features...")

        # Preprocess the DataFrame to extract synsets
        syns = self.preprocessor.preprocess_df(df, 'tokenise_noPunct_lowercase_POS_lemma_noStop_synset')

        def filter_synsets_by_pos(synsets, pos):
            """Filter synsets by part of speech."""
            return {s for s in synsets if s.pos() == pos}

        prefixes = ["all_all_", "all_verb_", "all_noun_", "all_adj_", "all_adv_", "best_all_", "best_verb_", "best_noun_", "best_adj_", "best_adv_"]
        pos_map = {
            "verb": nltk.corpus.wordnet.VERB,
            "noun": nltk.corpus.wordnet.NOUN,
            "adj": nltk.corpus.wordnet.ADJ,
            "adv": nltk.corpus.wordnet.ADV,
        }

        results = {prefix + metric: [] for prefix in prefixes for metric in ["shared_synsets_count", "shared_synsets_ratio", "avg_synset_similarity", "max_synset_similarity"]}

        total = len(syns)
        for i, (syns1_list, syns2_list) in enumerate(syns):
            # Flatten synsets for "all" and "best" calculations
            all_synsets1 = {synset for sublist in syns1_list for synset in sublist}
            all_synsets2 = {synset for sublist in syns2_list for synset in sublist}
            best_synsets1 = {sublist[0] for sublist in syns1_list if sublist}
            best_synsets2 = {sublist[0] for sublist in syns2_list if sublist}

            # Compute statistics for "all" synsets
            all_stats = self.compute_synset_distances(all_synsets1, all_synsets2)
            for metric, value in all_stats.items():
                results[f"all_all_{metric}"].append(value)

            # Compute statistics for specific POS categories
            for pos_name, pos in pos_map.items():
                filtered_syns1 = filter_synsets_by_pos(all_synsets1, pos)
                filtered_syns2 = filter_synsets_by_pos(all_synsets2, pos)
                pos_stats = self.compute_synset_distances(filtered_syns1, filtered_syns2)
                for metric, value in pos_stats.items():
                    results[f"all_{pos_name}_{metric}"].append(value)

            # Compute statistics for "best" synsets
            best_stats = self.compute_synset_distances(best_synsets1, best_synsets2)
            for metric, value in best_stats.items():
                results[f"best_all_{metric}"].append(value)

            # Compute statistics for specific POS categories for "best"
            for pos_name, pos in pos_map.items():
                filtered_syns1 = filter_synsets_by_pos(best_synsets1, pos)
                filtered_syns2 = filter_synsets_by_pos(best_synsets2, pos)
                pos_stats = self.compute_synset_distances(filtered_syns1, filtered_syns2)
                for metric, value in pos_stats.items():
                    results[f"best_{pos_name}_{metric}"].append(value)

            # Progress tracking
            if i % 10 == 0:
                print(f"\rProcessed {i} of {total} rows ({i * 100 / total:.1f}%)", end='', flush=True)

        print(f"\rProcessed {i + 1} of {total} rows (100%)      ")

        # Assign computed results to the DataFrame
        for column, values in results.items():
            df[column] = values


    def add_lemma_statistics(self, df: DataFrame) -> None:
        """
        Add lemma-based features to the DataFrame.
        """
        print("Adding lemma based features...")
        lemmas = self.preprocessor.preprocess_df(df, 'tokenise_noPunct_lowercase_noStop_lemma')

        relational_features = {
            'lemma_diversity': [],
            'shared_lemmas_ratio': [],
            'avg_lemma_similarity': [],
            'max_lemma_similarity': [],
            'lemma_jackard_similarity': [],
            'shared_lemma_count': [],
            'dice_coefficient': [],
            'lemma_bigram_overlap': [],
            'lemma_lcs_length': [],
            'lemma_edit_distance': [],
            'proportion_s1_in_s2': [],
            'proportion_s2_in_s1': [],
            'lemma_position_similarity': [],
        }
    
        def jaccard_similarity(lemma1, lemma2):
            set1 = set(lemma1)
            set2 = set(lemma2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union != 0 else 0
        
        def compute_lcs_length(s1: List[str], s2: List[str]) -> int:
            """
            Compute Longest Common Subsequence (LCS) length.
            """
            matcher = SequenceMatcher(None, s1, s2)
            return sum(block.size for block in matcher.get_matching_blocks())
        
        def compute_position_similarity(s1: List[str], s2: List[str]) -> float:
            """
            Compute position similarity for shared lemmas.
            """
            positions = [
                1 - abs(s1.index(lemma) - s2.index(lemma)) / max(len(s1), len(s2))
                for lemma in set(s1) & set(s2)
            ]
            return np.mean(positions) if positions else 0

        for pair in lemmas:
            s1_lemmas = [lemma for lemma in pair[0] if lemma]
            s2_lemmas = [lemma for lemma in pair[1] if lemma]

            # Lemma diversity
            total_unique_lemmas = len(set(s1_lemmas).union(set(s2_lemmas)))
            relational_features['lemma_diversity'].append(total_unique_lemmas)

            # Shared lemma ratio (normalized by total unique lemmas)
            shared_lemmas = set(s1_lemmas).intersection(set(s2_lemmas))
            shared_ratio = len(shared_lemmas) / total_unique_lemmas if total_unique_lemmas > 0 else 0
            relational_features['shared_lemmas_ratio'].append(shared_ratio)

            # Jaccard similarity
            relational_features['lemma_jackard_similarity'].append(1 - jaccard_distance(set(s1_lemmas), set(s2_lemmas)))

            similarities = [
                jaccard_similarity(set(lemma1), set(lemma2)) for lemma1 in s1_lemmas for lemma2 in s2_lemmas
            ]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            max_similarity = max(similarities) if similarities else 0
            relational_features['avg_lemma_similarity'].append(avg_similarity)
            relational_features['max_lemma_similarity'].append(max_similarity)

            # Shared lemma count
            shared_lemma_count = len(shared_lemmas)
            relational_features['shared_lemma_count'].append(shared_lemma_count)

            # Dice coefficient
            dice_coefficient = (2 * len(shared_lemmas)) / (len(set(s1_lemmas)) + len(set(s2_lemmas))) if (len(set(s1_lemmas)) + len(set(s2_lemmas))) > 0 else 0
            relational_features['dice_coefficient'].append(dice_coefficient)

            s1_bigrams = set(bigrams(s1_lemmas))
            s2_bigrams = set(bigrams(s2_lemmas))
            bigram_overlap = len(s1_bigrams & s2_bigrams) / len(s1_bigrams | s2_bigrams) if len(s1_bigrams | s2_bigrams) > 0 else 0
            relational_features['lemma_bigram_overlap'].append(bigram_overlap)

            # Longest Common Subsequence (LCS) length
            lcs_length = compute_lcs_length(s1_lemmas, s2_lemmas)
            relational_features['lemma_lcs_length'].append(lcs_length)

            # Lemma edit distance
            lemma_edit_distance = nltk.edit_distance(s1_lemmas, s2_lemmas)
            relational_features['lemma_edit_distance'].append(lemma_edit_distance)

            # Proportion of shared lemmas
            proportion_s1_in_s2 = len(shared_lemmas) / len(set(s1_lemmas)) if len(set(s1_lemmas)) > 0 else 0
            proportion_s2_in_s1 = len(shared_lemmas) / len(set(s2_lemmas)) if len(set(s2_lemmas)) > 0 else 0
            relational_features['proportion_s1_in_s2'].append(proportion_s1_in_s2)
            relational_features['proportion_s2_in_s1'].append(proportion_s2_in_s1)

            # Lemma Position similarity
            position_similarity = compute_position_similarity(s1_lemmas, s2_lemmas)
            relational_features['lemma_position_similarity'].append(position_similarity)

        for key, values in relational_features.items():
            df[key] = values