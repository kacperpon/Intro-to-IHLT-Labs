def add_dummy_features(df):
    df['s1_len'] = df['s1'].apply(len)
    df['s2_len'] = df['s2'].apply(len)
    df['s1_words'] = df['s1'].apply(lambda x: len(x.split()))
    df['s2_words'] = df['s2'].apply(lambda x: len(x.split()))
    df['dif_len'] = df.apply(lambda row: abs(row['s1_len'] - row['s2_len']), axis=1)
    df['dif_words'] = df.apply(lambda row: abs(row['s1_words'] - row['s2_words']), axis=1)
    return df