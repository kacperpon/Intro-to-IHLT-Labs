from functools import lru_cache
from itertools import product
from typing import List
from pandas import DataFrame
from utils.preprocessor import Preprocessor
import nltk
from nltk.util import bigrams
from difflib import SequenceMatcher

class FeatureExtractor:
    NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
    VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    ADJECTIVES = ['JJ', 'JJR', 'JJS']
    ADVERBS = ['RB', 'RBR', 'RBS']
    VERBS_PRESENT = ['VBG', 'VBP', 'VBZ']
    VERBS_PAST = ['VBD', 'VBN']
    VERBS_OTHER = ['VB']

    def __init__(self):
        self.preprocessor = Preprocessor()

    def add_POS_statistics(self, df):
        print("Adding POS based features...")
        # Create a preprocessor object
        preprop = Preprocessor()
        # Preprocess and extract POS tags for each sentence
        pos = preprop.preprocess_df(df, 'lowercase_noPunct_tokenize_onlyWords_noStop_POS')

        # relational_features = {
        #     'dif_n_words': [],
        #     'dif_n_verbs': [],
        #     'dif_n_nouns': [],
        #     'dif_n_adjectives': [],
        #     'dif_n_adverbs': []
        # }

        # Compute relational statistics for each sentence pair
        # for pair in pos:
        #     s1 = pair[0]  
        #     s2 = pair[1]
            
        #     # Relational features: differences
        #     relational_features['dif_n_words'].append(len(s1) - len(s2))
        #     relational_features['dif_n_verbs'].append(
        #         len([word for word in s1 if word[1] in VERBS]) - len([word for word in s2 if word[1] in VERBS])
        #     )
        #     relational_features['dif_n_nouns'].append(
        #         len([word for word in s1 if word[1] in NOUNS]) - len([word for word in s2 if word[1] in NOUNS])
        #     )
        #     relational_features['dif_n_adjectives'].append(
        #         len([word for word in s1 if word[1] in ADJECTIVES]) - len([word for word in s2 if word[1] in ADJECTIVES])
        #     )
        #     relational_features['dif_n_adverbs'].append(
        #         len([word for word in s1 if word[1] in ADVERBS]) - len([word for word in s2 if word[1] in ADVERBS])
        #     )

        # for feature_name, values in relational_features.items():
        #     df[feature_name] = values


        # Add POS statistics
        for i in range(len(pos)):
            # Number of words
            df.loc[i, 's1_n_words'] = len(pos[i][0])
            df.loc[i, 's2_n_words'] = len(pos[i][1])
            # Number of verbs
            df.loc[i, 's1_n_verbs_tot'] = len([word for word in pos[i][0] if word[1] in self.VERBS])
            df.loc[i, 's2_n_verbs_tot'] = len([word for word in pos[i][1] if word[1] in self.VERBS])
            # Verbs in present
            df.loc[i, 's1_n_verbs_pres'] = len([word for word in pos[i][0] if word[1] in self.VERBS_PRESENT])
            df.loc[i, 's2_n_verbs_pres'] = len([word for word in pos[i][1] if word[1] in self.VERBS_PRESENT])
            # Verbs in present
            df.loc[i, 's1_n_verbs_past'] = len([word for word in pos[i][0] if word[1] in self.VERBS_PAST])
            df.loc[i, 's2_n_verbs_past'] = len([word for word in pos[i][1] if word[1] in self.VERBS_PAST])
            # Number of nouns
            df.loc[i, 's1_n_nouns'] = len([word for word in pos[i][0] if word[1] in self.NOUNS])
            df.loc[i, 's2_n_nouns'] = len([word for word in pos[i][1] if word[1] in self.NOUNS])
            # Number of adjectives
            df.loc[i, 's1_n_adjectives'] = len([word for word in pos[i][0] if word[1] in self.ADJECTIVES])
            df.loc[i, 's2_n_adjectives'] = len([word for word in pos[i][1] if word[1] in self.ADJECTIVES])
            # Number of adverbs
            df.loc[i, 's1_n_adverbs'] = len([word for word in pos[i][0] if word[1] in self.ADVERBS])
            df.loc[i, 's2_n_adverbs'] = len([word for word in pos[i][1] if word[1] in self.ADVERBS])

        # Compute also the differences
        df['dif_n_words'] = df['s1_n_words'] - df['s2_n_words']
        df['dif_n_verbs_tot'] = df['s1_n_verbs_tot'] - df['s2_n_verbs_tot']
        df['dif_n_verbs_pres'] = df['s1_n_verbs_pres'] - df['s2_n_verbs_pres']
        df['dif_n_verbs_past'] = df['s1_n_verbs_past'] - df['s2_n_verbs_past']
        df['dif_n_nouns'] = df['s1_n_nouns'] - df['s2_n_nouns']
        df['dif_n_adjectives'] = df['s1_n_adjectives'] - df['s2_n_adjectives']
        df['dif_n_adverbs'] = df['s1_n_adverbs'] - df['s2_n_adverbs']


    def add_synset_statistics(self, df: DataFrame):
        preprop = Preprocessor()

        # Preprocess the DataFrame to extract synsets
        pos = preprop.preprocess_df(df, 'tokenise_noPunct_lowercase_POS_lemma_noStop_synset')

        relational_features = {
            'shared_synsets_count': [],
            'shared_synsets_ratio': [],
            'avg_synset_similarity': [],
            'max_synset_similarity': [],
        }

        @lru_cache(maxsize=None)
        def cached_wup_similarity(syn1, syn2):
            return syn1.wup_similarity(syn2)

        for pair in pos:

            print(pair)
            s1_synsets = [synset_list[:3] for synset_list in pair[0] if synset_list]
            s2_synsets = [synset_list[:3] for synset_list in pair[1] if synset_list]

            # Flatten and clean synsets for both sentences
            s1_synsets = [synset for synset_list in pair[0] for synset in synset_list]  # Flatten list of lists
            s2_synsets = [synset for synset_list in pair[1] for synset in synset_list]  # Flatten list of lists

            # Commenting this because it's very dangerous: if this is executed, two appends will be done for the same sentence pair, 
            # so the length will not match with the dataset!
            # Handle empty synset lists
            # if not s1_synsets or not s2_synsets:
            #     relational_features['shared_synsets_count'].append(0)
            #     relational_features['shared_synsets_ratio'].append(0)
            #     relational_features['avg_synset_similarity'].append(0)
            #     relational_features['max_synset_similarity'].append(0)
            #     continue

            # Shared synsets
            shared_synsets = set(s1_synsets).intersection(set(s2_synsets))
            relational_features['shared_synsets_count'].append(len(shared_synsets))

            # Shared synset ratio (normalized by total unique synsets)
            total_unique_synsets = len(set(s1_synsets).union(set(s2_synsets)))
            shared_ratio = len(shared_synsets) / total_unique_synsets if total_unique_synsets > 0 else 0
            relational_features['shared_synsets_ratio'].append(shared_ratio)

            similarities = [
                cached_wup_similarity(syn1, syn2)
                for syn1, syn2 in product(s1_synsets, s2_synsets)
                if cached_wup_similarity(syn1, syn2) is not None
            ]

            # Calculate average and maximum similarities
            if similarities:
                relational_features['avg_synset_similarity'].append(sum(similarities) / len(similarities))
                relational_features['max_synset_similarity'].append(max(similarities))
            else:
                relational_features['avg_synset_similarity'].append(0)
                relational_features['max_synset_similarity'].append(0)

        # Add the relational features to the DataFrame
        for feature_name, values in relational_features.items():
            df[feature_name] = values

    @lru_cache(maxsize=None)
    def cached_wup_similarity(syn1, syn2):
        return syn1.wup_similarity(syn2)

    def compute_sysnsets_distances(self, df, row, prefix, synsets1, synsets2):
        shared_synsets = synsets1.intersection(synsets2)
        total_unique_synsets = len(set(synsets1).union(set(synsets2)))
        shared_ratio = len(shared_synsets) / total_unique_synsets if total_unique_synsets > 0 else -1

        similarities = [
            self.cached_wup_similarity(syn1, syn2)
            for syn1, syn2 in product(synsets1, synsets2)
            if syn1.pos() == syn2.pos() and self.cached_wup_similarity(syn1, syn2) is not None
        ]

        df.loc[row, prefix + 'shared_synsets_count'] = len(shared_synsets)
        df.loc[row, prefix + 'shared_synsets_ratio'] = shared_ratio
        df.loc[row, prefix + 'avg_synset_similarity'] = sum(similarities) / len(similarities) if similarities else -1
        df.loc[row, prefix + 'max_synset_similarity'] = max(similarities) if similarities else -1

    def add_synset_statistics_ext(self, df: DataFrame):
        print("Adding synset based features...")

        preprop = Preprocessor()

        # Preprocess the DataFrame to extract synsets
        syns = preprop.preprocess_df(df, 'tokenise_noPunct_lowercase_POS_lemma_noStop_synset')

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

    # Lemma-based techniques
    def add_lemma_statistics(self, df: DataFrame):
        preprop = Preprocessor()

        # Preprocess the DataFrame to extract lemmas
        lemmas = preprop.preprocess_df(df, 'tokenise_noPunct_lowercase_noStop_lemma')

        relational_features = {
            'lemma_diversity': [],
            'shared_lemmas_ratio': [],
            'avg_lemma_similarity': [],
            'max_lemma_similarity': [],
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
            matcher = SequenceMatcher(None, s1, s2)
            return sum(block.size for block in matcher.get_matching_blocks())
        
        def compute_position_similarity(s1: List[str], s2: List[str]) -> float:
            positions = []
            for lemma in set(s1) & set(s2):
                pos_s1 = s1.index(lemma)
                pos_s2 = s2.index(lemma)
                positions.append(1 - abs(pos_s1 - pos_s2) / max(len(s1), len(s2)))
            return sum(positions) / len(positions) if positions else 0

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

            lemma_edit_distance = nltk.edit_distance(s1_lemmas, s2_lemmas)
            relational_features['lemma_edit_distance'].append(lemma_edit_distance)

            # Proportion of lemmas
            proportion_s1_in_s2 = len(shared_lemmas) / len(set(s1_lemmas)) if len(set(s1_lemmas)) > 0 else 0
            proportion_s2_in_s1 = len(shared_lemmas) / len(set(s2_lemmas)) if len(set(s2_lemmas)) > 0 else 0
            relational_features['proportion_s1_in_s2'].append(proportion_s1_in_s2)
            relational_features['proportion_s2_in_s1'].append(proportion_s2_in_s1)

            # Position similarity
            position_similarity = compute_position_similarity(s1_lemmas, s2_lemmas)
            relational_features['lemma_position_similarity'].append(position_similarity)

        for key, values in relational_features.items():
            df[key] = values