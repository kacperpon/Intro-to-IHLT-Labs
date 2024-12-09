import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from nltk import pos_tag
from nltk.corpus import wordnet

class Preprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
    
        self.pipelines = {
            'lowercase_noPunct_tokenize_onlyWords_noStop_POS': [
                self.lowercase,
                self.remove_punctuation,
                self.tokenize,
                self.keep_only_alpha,
                self.remove_stopwords,
                self.POS_tag
            ],
            'lowercase_tokenize_POS': [
                self.lowercase,
                self.tokenize,
                self.POS_tag
            ],
            'tokenise_noPunct_lowercase_POS_lemma_noStop_synset': [
                self.lowercase,
                self.remove_punctuation,
                self.tokenize,
                self.remove_stopwords,
                self.POS_tag,
                self.lemmatize_pos,
                self.synset
            ],
            'tokenise_noPunct_lowercase_noStop_lemma': [
                self.lowercase,
                self.remove_punctuation,
                self.tokenize,
                self.remove_stopwords,
                self.lemmatize
            ],
        }

    def lowercase(self, text: str) -> str:
        """
        Convert the text to lowercase.
        """
        return text.lower()
    
    def tokenize(self, text: str) -> list:
        """
        Tokenize the text.
        """
        return nltk.word_tokenize(text)

    def get_wordnet_pos(self, treebank_tag):
        """
        Convert treebank tags to wordnet tags.
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    def lemmatize_pos(self, tokens: list) -> list:
        """
        Lemmatize the tokens with POS tags.
        """
        lemmatizer = nltk.WordNetLemmatizer()
        lemmatized_tokens = []
        for token, tag in tokens:
            wordnet_pos = self.get_wordnet_pos(tag) or wordnet.NOUN  # Default to NOUN if no mapping
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos=wordnet_pos))
        return lemmatized_tokens
    
    def lemmatize(self, tokens: list) -> list:
        """
        Lemmatize the tokens.
        """
        lemmatizer = nltk.WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from the text.
        """
        return re.sub(r'[^\w\s]', '', text)

    def keep_only_alpha(self, tokens: list) -> list:
        """
        Keep only alphabetic characters in the tokens.
        """
        return [w for w in tokens if w.isalpha()] 

    def remove_stopwords(self, tokens: list) -> list:
        """
        Remove stopwords from the tokens.
        """
        return [w for w in tokens if w not in self.stopwords]

    def POS_tag(self, tokens: list) -> list:
        """
        Get the POS tags of the tokens.
        """
        return nltk.pos_tag(tokens)
    
    def synset(self, tokens: list) -> list:
        """
        Get the synsets of the tokens.
        """
        return [nltk.corpus.wordnet.synsets(token) for token in tokens]

    def add_pipeline(self, name: str, steps: list):
        """
        Add a new preprocessing pipeline.
        """
        self.pipelines[name] = steps

    def preprocess(self, text: str, pipeline_name: str) -> str:
        """
        Apply the specified preprocessing pipeline to the text.
        """
        pipeline = self.pipelines.get(pipeline_name)

        if pipeline is None:
            raise ValueError(f"Pipeline '{pipeline_name}' not found.")
    
        for step in pipeline:
            text = step(text)

        return text
    
    def preprocess_df(self, df: pd.DataFrame, pipeline_name: str) -> any:
        """
        Preprocess content in the DataFrame based on the pipeline mapping.
        Returns array of tuples of two strings (sentence1, sentence2)
        """
        result = []

        for _, row in df.iterrows():
            processed_s1 = self.preprocess(row['s1'], pipeline_name)
            processed_s2 = self.preprocess(row['s2'], pipeline_name)
            result.append((processed_s1, processed_s2))

        return result
