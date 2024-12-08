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

        def _compute_pos_statistics(self, pos_tags: List[Tuple[str, str]], pos_set: Set[str]) -> int:
            """
            Count POS occurrences in a tagged sentence.
            """
            return len([word for word, pos in pos_tags if pos in pos_set])

    def add_POS_statistics(self, df):
        print("Adding POS based features...")

        pos = self.preprocessor.preprocess_df(df, 'lowercase_tokenize_POS')

        # Add POS statistics
        for i in range(len(pos)):
            # Number of words
            df.loc[i, 's1_n_words'] = len(pos[i][0])
            df.loc[i, 's2_n_words'] = len(pos[i][1])

            # Number of verbs
            verbs1 = [word for word in pos[i][0] if word[1] in self.VERBS]
            verbs2 = [word for word in pos[i][1] if word[1] in self.VERBS]

            df.loc[i, 's1_n_verbs_tot'] = len(verbs1)
            df.loc[i, 's2_n_verbs_tot'] = len(verbs2)

            # Verbs in present
            df.loc[i, 's1_n_verbs_pres'] = len([word for word in pos[i][0] if word[1] in self.VERBS_PRESENT])
            df.loc[i, 's2_n_verbs_pres'] = len([word for word in pos[i][1] if word[1] in self.VERBS_PRESENT])

            # Verbs in present
            df.loc[i, 's1_n_verbs_past'] = len([word for word in pos[i][0] if word[1] in self.VERBS_PAST])
            df.loc[i, 's2_n_verbs_past'] = len([word for word in pos[i][1] if word[1] in self.VERBS_PAST])

            # Number of nouns
            nouns1 = [word for word in pos[i][0] if word[1] in self.NOUNS]
            nouns2 = [word for word in pos[i][1] if word[1] in self.NOUNS]
            df.loc[i, 's1_n_nouns'] = len(nouns1)
            df.loc[i, 's2_n_nouns'] = len(nouns2)
            
            # Number of adjectives
            adj1 = [word for word in pos[i][0] if word[1] in self.ADJECTIVES]
            adj2 = [word for word in pos[i][1] if word[1] in self.ADJECTIVES]
            df.loc[i, 's1_n_adjectives'] = len(adj1)
            df.loc[i, 's2_n_adjectives'] = len(adj2)

            # Number of adverbs
            adv1 = [word for word in pos[i][0] if word[1] in self.ADVERBS]
            adv2 = [word for word in pos[i][1] if word[1] in self.ADVERBS]
            df.loc[i, 's1_n_adverbs'] = len(adv1)
            df.loc[i, 's2_n_adverbs'] = len(adv2)

        # Compute also the differences
        df['dif_n_words'] = df['s1_n_words'] - df['s2_n_words']
        df['dif_n_verbs_tot'] = df['s1_n_verbs_tot'] - df['s2_n_verbs_tot']
        df['dif_n_verbs_pres'] = df['s1_n_verbs_pres'] - df['s2_n_verbs_pres']
        df['dif_n_verbs_past'] = df['s1_n_verbs_past'] - df['s2_n_verbs_past']
        df['dif_n_nouns'] = df['s1_n_nouns'] - df['s2_n_nouns']
        df['dif_n_adjectives'] = df['s1_n_adjectives'] - df['s2_n_adjectives']
        df['dif_n_adverbs'] = df['s1_n_adverbs'] - df['s2_n_adverbs']

        # Jaccard distance in all words
        df['jaccard_all_words'] = 1 - jaccard_distance(set(pos[i][0]), set(pos[i][1])) 
        df['jaccard_verbs'] = 1 - jaccard_distance(set(verbs1), set(verbs2)) if len(verbs1) > 0 and len(verbs2) > 0 else 0
        df['jaccard_nouns'] = 1 - jaccard_distance(set(nouns1), set(nouns2)) if len(nouns1) > 0 and len(nouns2) > 0 else 0
        df['jaccard_adjectives'] = 1 - jaccard_distance(set(adj1), set(adj2)) if len(adj1) > 0 and len(adj2) > 0 else 0
        df['jaccard_adverbs'] = 1 - jaccard_distance(set(adv1), set(adv2)) if len(adv1) > 0 and len(adv2) > 0 else 0


    def compute_sysnsets_distances(self, df, row, prefix, synsets1, synsets2):
        shared_synsets = synsets1.intersection(synsets2)
        total_unique_synsets = len(set(synsets1).union(set(synsets2)))
        shared_ratio = len(shared_synsets) / total_unique_synsets if total_unique_synsets > 0 else 0

        similarities = [
            cached_wup_similarity(syn1, syn2)
            for syn1, syn2 in product(synsets1, synsets2)
            if syn1.pos() == syn2.pos() and cached_wup_similarity(syn1, syn2) is not None
        ]

        df.loc[row, prefix + 'shared_synsets_count'] = len(shared_synsets)
        df.loc[row, prefix + 'shared_synsets_ratio'] = shared_ratio
        if similarities:
            df.loc[row, prefix + 'avg_synset_similarity'] = sum(similarities) / len(similarities)
            df.loc[row, prefix + 'max_synset_similarity'] = max(similarities)
        elif len(synsets1) == 0 and len(synsets2) == 0:
            df.loc[row, prefix + 'avg_synset_similarity'] = 1
            df.loc[row, prefix + 'max_synset_similarity'] = 1
        else:
            df.loc[row, prefix + 'avg_synset_similarity'] = 0
            df.loc[row, prefix + 'max_synset_similarity'] = 0


    def add_synset_statistics_ext(self, df: DataFrame):
        print("Adding synset based features...")

        # Preprocess the DataFrame to extract synsets
        syns = self.preprocessor.preprocess_df(df, 'tokenise_noPunct_lowercase_POS_lemma_noStop_synset')

        total = len(syns)
        for i in range(total):
            # All synsets statistics
            s1 = {synset for sublist in syns[i][0] for synset in sublist}
            s2 = {synset for sublist in syns[i][1] for synset in sublist}
            self.compute_sysnsets_distances(df, i, 'all_all_', s1, s2)
            self.compute_sysnsets_distances(df, i, 'all_verb_', {s for s in s1 if s.pos() == nltk.corpus.wordnet.VERB}, {s for s in s2 if s.pos() == nltk.corpus.wordnet.VERB})
            self.compute_sysnsets_distances(df, i, 'all_noun_', {s for s in s1 if s.pos() == nltk.corpus.wordnet.NOUN}, {s for s in s2 if s.pos() == nltk.corpus.wordnet.NOUN})
            self.compute_sysnsets_distances(df, i, 'all_adj_', {s for s in s1 if s.pos() == nltk.corpus.wordnet.ADJ}, {s for s in s2 if s.pos() == nltk.corpus.wordnet.ADJ})
            self.compute_sysnsets_distances(df, i, 'all_adv_', {s for s in s1 if s.pos() == nltk.corpus.wordnet.ADV}, {s for s in s2 if s.pos() == nltk.corpus.wordnet.ADV})
        
            # Most common sysnset statistics
            s1 = {synset[0] for synset in syns[i][0] if len(synset) > 0}
            s2 = {synset[0] for synset in syns[i][1] if len(synset) > 0}
            self.compute_sysnsets_distances(df, i, 'best_all_', s1, s2)
            self.compute_sysnsets_distances(df, i, 'best_verb_', {s for s in s1 if s.pos() == nltk.corpus.wordnet.VERB}, {s for s in s2 if s.pos() == nltk.corpus.wordnet.VERB})
            self.compute_sysnsets_distances(df, i, 'best_noun_', {s for s in s1 if s.pos() == nltk.corpus.wordnet.NOUN}, {s for s in s2 if s.pos() == nltk.corpus.wordnet.NOUN})
            self.compute_sysnsets_distances(df, i, 'best_adj_', {s for s in s1 if s.pos() == nltk.corpus.wordnet.ADJ}, {s for s in s2 if s.pos() == nltk.corpus.wordnet.ADJ})
            self.compute_sysnsets_distances(df, i, 'best_adv_', {s for s in s1 if s.pos() == nltk.corpus.wordnet.ADV}, {s for s in s2 if s.pos() == nltk.corpus.wordnet.ADV})
            if i % 10 == 0:
                print(f"\rProcessed {i} of {total} rows ({i*100/total:.1f}%)", end='', flush=True)
        
        print(f"\rProcessed {i + 1}/{len(syns)} rows (100.0%)    ")


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
            jaccard_similarity = 1 - jaccard_distance(set(s1_lemmas), set(s2_lemmas))
            relational_features['lemma_jackard_similarity'].append(jaccard_similarity)

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