import os
import pandas as pd

def load_sentence_files(path: str) -> pd.DataFrame:
    """
    Load sentence files from the given path into a DataFrame.
    """
    files = [f for f in os.listdir(path) if 'STS.gs' not in f and '.pl' not in f and 'readme' not in f]

    file_contents = []
    
    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            content = f.read()

        file_contents.append({'filename': file, 'content': content})
    return pd.DataFrame(file_contents)


def save_preprocessed_files(sentence_files: pd.DataFrame, target_path: str):
    """
    Save preprocessed DataFrame to files in the specified target path.
    """
    os.makedirs(target_path, exist_ok=True)

    for _, row in sentence_files.iterrows():
        file_name = row['filename']
        content = row['content']

        target_file = os.path.join(target_path, file_name)
        with open(target_file, 'w', encoding='utf-8') as f:

            f.write(content)
