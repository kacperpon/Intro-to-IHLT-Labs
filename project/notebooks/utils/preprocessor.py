import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from nltk import pos_tag

class Preprocessor:
    def __init__(self):
        # Different preprocessing pipelines, modify if required
        self.pipelines = {
            'semantic': [self.lowercase, self.tokenize, self.lemmatize],
            'jaccard': [self.lowercase, self.remove_punctuation, self.tokenize],
            'lowest_common_subsequence': [self.lowercase],
            #'pos_tag_overlap': [self.lowercase, self.POS_tag],
            'sentence_length_ratio': [self.tokenize],
            'default': [self.lowercase],

            # Alternative naming for pipelines
            'lowercase_noPunct_tokenize_onlyWords_noStop_POS': [self.lowercase, self.remove_punctuation, self.tokenize, self.remove_no_words, self.remove_stopwords, self.POS_tag],
        }
        self.stopwords = set(stopwords.words('english'))

    def lowercase(self, text: str) -> str:
        return text.lower()

    def tokenize(self, text: str) -> list:
        return nltk.word_tokenize(text)

    def lemmatize(self, tokens: list) -> list:
        lemmatizer = nltk.WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', text)

    def remove_no_words(self, tokens: list) -> list:
        return [w for w in tokens if w.isalpha()] 

    def remove_stopwords(self, tokens: list) -> list:
        return [w for w in tokens if w not in self.stopwords]

    def POS_tag(self, tokens: list) -> list:
        return nltk.pos_tag(tokens)

    def add_pipeline(self, name: str, steps: list):
        """
        Add a new preprocessing pipeline.
        """
        self.pipelines[name] = steps

    def preprocess(self, text: str, pipeline_name: str) -> str:
        """
        Apply the specified preprocessing pipeline to the text.
        """
        pipeline = self.pipelines.get(pipeline_name, self.pipelines['default'])
        for step in pipeline:
            text = step(text)
        return text

    def preprocess_files(self, sentence_files: pd.DataFrame, pipeline_mapping: dict) -> pd.DataFrame:
        """
        Preprocess content in the DataFrame based on the pipeline mapping.
        """
        for idx, row in sentence_files.iterrows():
            filename = row['filename']

            # Determine pipeline based on file type or filename pattern
            pipeline_name = pipeline_mapping.get(filename.split('_')[0], 'default')

            # Apply preprocessing
            content = row['content']
            sentence_files.at[idx, 'content'] = self.preprocess(content, pipeline_name)

        return sentence_files
    
    def preprocess_df(self, df: pd.DataFrame, pipeline_name: str) -> any:
        """
        Preprocess content in the DataFrame based on the pipeline mapping.
        Returns array of tuples of two strings (sentence1, sentence2)
        """
        ret = []
        for idx, row in df.iterrows():
            ret.append((self.preprocess(row['s1'], pipeline_name), self.preprocess(row['s2'], pipeline_name)))

        return ret
    

