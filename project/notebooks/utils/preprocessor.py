import re
import nltk
import pandas as pd

class Preprocessor:
    def __init__(self):
        # Different preprocessing pipelines, modify if required
        self.pipelines = {
            'semantic': [self.lowercase, self.tokenize, self.lemmatize],
            'lexical': [self.lowercase, self.remove_punctuation, self.tokenize],
            'structural': [self.lowercase, self.tokenize],
            'default': [self.lowercase]
        }

    def lowercase(self, text: str) -> str:
        return text.lower()

    def tokenize(self, text: str) -> list:
        return nltk.word_tokenize(text)

    def lemmatize(self, tokens: list) -> list:
        lemmatizer = nltk.WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    def remove_punctuation(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', text)

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
        :param sentence_files: DataFrame containing sentence files.
        :param pipeline_mapping: Mapping of file types or features to pipeline names.
        :return: Preprocessed DataFrame.
        """
        for idx, row in sentence_files.iterrows():
            filename = row['filename']

            # Determine pipeline based on file type or filename pattern
            pipeline_name = pipeline_mapping.get(filename.split('_')[0], 'default')

            # Apply preprocessing
            content = row['content']
            sentence_files.at[idx, 'content'] = self.preprocess(content, pipeline_name)

        return sentence_files
