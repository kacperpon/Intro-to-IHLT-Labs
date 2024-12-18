import csv
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import re
import datetime
import os
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

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

def evaluate_rf_model(model_trainer, df_train, df_test, features, target_column, save_path, pred_col, n_iterations=10, save_results=True):
    """
    Evaluate a Random Forest model using Pearson correlation.
    """
    # Find the best parameter combination and train the model
    best_model, best_params, feature_importance_figure = model_trainer.train_RF(df_train, features, target_column)
    
    # Predict on test data and calculate single iteration correlation
    df_test['predicted'] = best_model.predict(df_test[features])
    single_iteration_correlation = pearsonr(df_test[target_column], df_test['predicted'])[0]
    print(f'\nPearson correlation for the best RF model: {single_iteration_correlation}')

    # Calculate regression metrics
    mse = mean_squared_error(df_test[target_column], df_test['predicted'])
    rmse = mse ** 0.5
    
    # Calculate mean Pearson correlation
    print(f"Computing mean Pearson correlation for {n_iterations} iterations...")
    total_correlation = 0
    correlations = []
    
    for _ in range(n_iterations):
        # Train a single RF model with the best parameters
        rf_model = model_trainer.train_single_RF(df_train, features, target_column, best_params)
        
        # Predict and calculate correlation
        df_test['predicted'] = rf_model.predict(df_test[features])
        iteration_correlation = pearsonr(df_test[target_column], df_test['predicted'])[0]
        correlations.append(iteration_correlation)
        total_correlation += iteration_correlation
    
    mean_correlation = total_correlation / n_iterations
    print(f'Mean Pearson correlation over {n_iterations} iterations: {mean_correlation}')
    
    if save_results:
        save_predictions(df_test, save_path, pred_col, feature_importance_figure)

    metrics = {
        "single_iteration_correlation": single_iteration_correlation,
        "mean_correlation": mean_correlation,
        "max_correlation": max(correlations),
        "min_correlation": min(correlations),
        "std_correlation": (sum((x - mean_correlation) ** 2 for x in correlations) / n_iterations) ** 0.5,
        "rmse": rmse,
    }
    
    return best_model, best_params, metrics, mean_correlation

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

def update_results_csv(results_file, model_name, feature_set, metrics, prediction_file):
    """
    Update the results.csv file with run details.
    """
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file is new (doesn't exist yet)
    is_new_file = not results_path.exists()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "Timestamp": timestamp,
        "Model": model_name,
        "Feature Set": feature_set,
        "RMSE": metrics.get("rmse", ""),
        "Mean Correlation": metrics.get("mean_correlation", ""),
        "Max Correlation": metrics.get("max_correlation", ""),
        "Standard Deivation of Correlations": metrics.get("std_correlation", ""),
        "Single Iteration Correlation": metrics.get("single_iteration_correlation", ""),
        "Prediction File": prediction_file,
    }

    with results_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if is_new_file:  # Write header if file is new
            writer.writeheader()
        writer.writerow(row)


def generate_plots_from_metrics(file_path, save_path="plots"):
    """
    Generate plots from metrics.csv to visualize performance metrics.
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_path)

    # Convert 'Timestamp' to datetime for time-based analysis
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Plot RMSE by Feature Set (bar plot)
    plt.figure(figsize=(12, 8))
    for feature_set in df['Feature Set'].unique():
        subset = df[df['Feature Set'] == feature_set]
        plt.bar(subset['Feature Set'], subset['RMSE'], label=feature_set)
    plt.xlabel('Feature Set')
    plt.ylabel('RMSE')
    plt.title('RMSE by Feature Set for Random Forest')
    plt.legend(title='Feature Set')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_path}/rf_rmse_by_feature_set.png")
    plt.close()

    # Mean Correlation by Feature Set (line plot with markers)
    plt.figure(figsize=(12, 8))
    for feature_set in df['Feature Set'].unique():
        subset = df[df['Feature Set'] == feature_set]
        plt.plot(subset['Feature Set'], subset['Mean Correlation'], marker='o', markersize=16, label=feature_set)
    plt.xlabel('Feature Set')
    plt.ylabel('Mean Correlation')
    plt.title('Mean Correlation by Feature Set for Random Forest')
    plt.legend(title='Feature Set')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_path}/rf_mean_correlation_by_feature_set.png")
    plt.close()

    # Correlation Standard Deviation (boxplot)
    plt.figure(figsize=(12, 8))
    df.boxplot(column='Standard Deivation of Correlations', by='Feature Set', grid=False, patch_artist=True, showmeans=True)
    plt.xlabel('Feature Set')
    plt.ylabel('Standard Deviation of Correlations')
    plt.title('Standard Deviation of Correlations by Feature Set for Random Forest')
    plt.suptitle('')  # Remove default boxplot title
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_path}/rf_std_correlation_by_feature_set.png")
    plt.close()

    print(f"Plots have been saved to {save_path}/")

