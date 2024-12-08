import csv
import pandas as pd
import re
import datetime
import os

def load_data(path_f, path_gs, files) -> pd.DataFrame:
    """
    Load data from files and return a DataFrame.
    """
    # Read first file
    dt = pd.read_csv(path_f + 'STS.input.' + files[0] + '.txt', sep='\t', quoting=csv.QUOTE_NONE, header=None, names=['s1', 's2'])
    dt['gs'] = pd.read_csv(path_gs + 'STS.gs.' + files[0] + '.txt', sep='\t', header=None, names=['gs'])
    dt['file'] = files[0]

    # Concatenate the rest of files
    for f in files[1:]:
        dt2 = pd.read_csv(path_f + 'STS.input.' + f + '.txt', sep='\t', quoting=csv.QUOTE_NONE, header=None, names=['s1', 's2'])
        dt2['gs'] = pd.read_csv(path_gs + 'STS.gs.' + f + '.txt', sep='\t', header=None, names=['gs'])
        dt2['file']=f
        dt = pd.concat([dt, dt2], ignore_index=True)
        
    return dt

def save_predictions(df, save_path, model_name):
    """
    Save predictions to a file.
    """
    os.makedirs(save_path, exist_ok=True)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    csv_file = os.path.join(save_path, f'{timestamp}_{model_name}_predicted_test_data.csv')
    df.to_csv(csv_file, index=False)
    print(f"Predicted data saved to CSV: {csv_file}")

    df_clean = clean_illegal_characters(df)
    excel_file = os.path.join(save_path, f'{timestamp}_{model_name}_predicted_test_data.xlsx')
    df_clean.to_excel(excel_file, index=False)
    print(f"Predicted data saved to Excel: {excel_file}")

def clean_illegal_characters(df):
    """
    Remove illegal characters from a DataFrame.
    """
    illegal_characters_re = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    
    def clean_value(value):
        if isinstance(value, str):
            return illegal_characters_re.sub('', value)
        return value
    
    return df.map(clean_value)