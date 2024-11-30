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
    # Add POS statistics
    for i in range(len(pos)):
        # Number of words
        df.loc[i, 's1_n_words'] = len(pos[i][0])
        df.loc[i, 's2_n_words'] = len(pos[i][1])
        # Number of verbs
        df.loc[i, 's1_n_verbs'] = len([word for word in pos[i][0] if word[1] in VERBS])
        df.loc[i, 's2_n_verbs'] = len([word for word in pos[i][1] if word[1] in VERBS])
        # Number of nouns
        df.loc[i, 's1_n_nouns'] = len([word for word in pos[i][0] if word[1] in NOUNS])
        df.loc[i, 's2_n_nouns'] = len([word for word in pos[i][1] if word[1] in NOUNS])
        # Number of adjectives
        df.loc[i, 's1_n_adjectives'] = len([word for word in pos[i][0] if word[1] in ADJECTIVES])
        df.loc[i, 's2_n_adjectives'] = len([word for word in pos[i][1] if word[1] in ADJECTIVES])
        # Number of adverbs
        df.loc[i, 's1_n_adverbs'] = len([word for word in pos[i][0] if word[1] in ADVERBS])
        df.loc[i, 's2_n_adverbs'] = len([word for word in pos[i][1] if word[1] in ADVERBS])
        # print(df.loc[i])
        # print("S1: ", pos[i][0])
        # print("S2: ", pos[i][1])

    # Compute also the differences
    df['dif_n_words'] = df['s1_n_words'] - df['s2_n_words']
    df['dif_n_verbs'] = df['s1_n_verbs'] - df['s2_n_verbs']
    df['dif_n_nouns'] = df['s1_n_nouns'] - df['s2_n_nouns']
    df['dif_n_adjectives'] = df['s1_n_adjectives'] - df['s2_n_adjectives']
    df['dif_n_adverbs'] = df['s1_n_adverbs'] - df['s2_n_adverbs']
    # print(df.loc[0])
