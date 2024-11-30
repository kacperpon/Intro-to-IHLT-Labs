from utils.preprocessor import Preprocessor

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
ADJECTIVES = ['JJ', 'JJR', 'JJS']
ADVERBS = ['RB', 'RBR', 'RBS']

def add_POS_statistics(df):
    # Create a preprocessor object
    preprop = Preprocessor()
    # Preprocess and extract POS tags for each sentence
    pos = preprop.preprocess_df(df, 'lowercase_noPunct_tokenize_onlyWords_noStop_POS')

    relational_features = {
        'dif_n_words': [],
        'dif_n_verbs': [],
        'dif_n_nouns': [],
        'dif_n_adjectives': [],
        'dif_n_adverbs': []
    }

    # Compute relational statistics for each sentence pair
    for pair in pos:
        s1 = pair[0]  
        s2 = pair[1]
        
        # Relational features: differences
        relational_features['dif_n_words'].append(len(s1) - len(s2))
        relational_features['dif_n_verbs'].append(
            len([word for word in s1 if word[1] in VERBS]) - len([word for word in s2 if word[1] in VERBS])
        )
        relational_features['dif_n_nouns'].append(
            len([word for word in s1 if word[1] in NOUNS]) - len([word for word in s2 if word[1] in NOUNS])
        )
        relational_features['dif_n_adjectives'].append(
            len([word for word in s1 if word[1] in ADJECTIVES]) - len([word for word in s2 if word[1] in ADJECTIVES])
        )
        relational_features['dif_n_adverbs'].append(
            len([word for word in s1 if word[1] in ADVERBS]) - len([word for word in s2 if word[1] in ADVERBS])
        )

    for feature_name, values in relational_features.items():
        df[feature_name] = values

    # NOTE: Changed this because (I think) we don't need to be storing the individual statistics in the dataframe,
    # Only the differences. We cannot use individual statistics for training.

    # Add POS statistics
    # for i in range(len(pos)):
    #     # Number of words
    #     df.loc[i, 's1_n_words'] = len(pos[i][0])
    #     df.loc[i, 's2_n_words'] = len(pos[i][1])
    #     # Number of verbs
    #     df.loc[i, 's1_n_verbs'] = len([word for word in pos[i][0] if word[1] in VERBS])
    #     df.loc[i, 's2_n_verbs'] = len([word for word in pos[i][1] if word[1] in VERBS])
    #     # Number of nouns
    #     df.loc[i, 's1_n_nouns'] = len([word for word in pos[i][0] if word[1] in NOUNS])
    #     df.loc[i, 's2_n_nouns'] = len([word for word in pos[i][1] if word[1] in NOUNS])
    #     # Number of adjectives
    #     df.loc[i, 's1_n_adjectives'] = len([word for word in pos[i][0] if word[1] in ADJECTIVES])
    #     df.loc[i, 's2_n_adjectives'] = len([word for word in pos[i][1] if word[1] in ADJECTIVES])
    #     # Number of adverbs
    #     df.loc[i, 's1_n_adverbs'] = len([word for word in pos[i][0] if word[1] in ADVERBS])
    #     df.loc[i, 's2_n_adverbs'] = len([word for word in pos[i][1] if word[1] in ADVERBS])
    #     # print(df.loc[i])
    #     # print("S1: ", pos[i][0])
    #     # print("S2: ", pos[i][1])

    # # Compute also the differences
    # df['dif_n_words'] = df['s1_n_words'] - df['s2_n_words']
    # df['dif_n_verbs'] = df['s1_n_verbs'] - df['s2_n_verbs']
    # df['dif_n_nouns'] = df['s1_n_nouns'] - df['s2_n_nouns']
    # df['dif_n_adjectives'] = df['s1_n_adjectives'] - df['s2_n_adjectives']
    # df['dif_n_adverbs'] = df['s1_n_adverbs'] - df['s2_n_adverbs']
    # # print(df.loc[0])
