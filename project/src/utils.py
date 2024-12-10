import csv
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import re
import datetime
import os
from scipy.stats import pearsonr

THRESHOLD = 0.8

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

def save_predictions(df, save_path, model_name, figure=None):
    """
    Save predictions to a file.
    """
    model_dir = Path(save_path) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save predictions to CSV
    csv_file = model_dir / f"{timestamp}_test_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"Predicted data saved to CSV: {csv_file}")

    # Save predictions to Excel
    df_clean = clean_illegal_characters(df)
    excel_file = model_dir / f"{timestamp}_test_data.xlsx"
    df_clean.to_excel(excel_file, index=False)

    if figure:
        graph_file = model_dir / f"{timestamp}_feature_importance.png"
        figure.savefig(graph_file, bbox_inches="tight")
        print(f"Feature importance graph saved as: {graph_file}")
        plt.close(figure)

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

def evaluate_rf_model(model_trainer, df_train, df_test, features, target_column, save_path, pred_col, n_iterations=10):
    """
    Evaluate a Random Forest model using Pearson correlation.
    """
    # Find the best parameter combination and train the model
    best_model, best_params, feature_importance_figure = model_trainer.train_RF(df_train, features, target_column)
    
    # Predict on test data and calculate single iteration correlation
    df_test['predicted'] = best_model.predict(df_test[features])
    single_iteration_correlation = pearsonr(df_test[target_column], df_test['predicted'])[0]
    
    print(f'\nPearson correlation for the best RF model: {single_iteration_correlation}')
    
    # Calculate mean Pearson correlation
    print(f"Computing mean Pearson correlation for {n_iterations} iterations...")
    total = 0
    
    for _ in range(n_iterations):
        # Train a single RF model with the best parameters
        rf_model = model_trainer.train_single_RF(df_train, features, target_column, best_params)
        
        # Predict and calculate correlation
        df_test['predicted'] = rf_model.predict(df_test[features])
        total += pearsonr(df_test[target_column], df_test['predicted'])[0]
    
    mean_correlation = total / n_iterations
    print(f'Mean Pearson correlation over {n_iterations} iterations: {mean_correlation}')
    
    save_predictions(df_test, save_path, pred_col, feature_importance_figure)
    
    return best_model, best_params, single_iteration_correlation, mean_correlation

def drop_highly_correlated_features(df, threshold=0.8) -> list:
    """
    Drop highly correlated features from a DataFrame.
    """
    features_to_drop = set()
    
    # Recompute the correlation matrix and variances
    correlation_matrix = df.corr()
    variances = df.var()

    for col in correlation_matrix.columns:
        for index in correlation_matrix.index:
            if col != index and abs(correlation_matrix[col][index]) > threshold:
                # Drop the feature with lower variance
                if col not in features_to_drop and index not in features_to_drop:  # Avoid re-checking
                    if variances[col] < variances[index]:
                        features_to_drop.add(col)
                    else:
                        features_to_drop.add(index)

    return list(features_to_drop)