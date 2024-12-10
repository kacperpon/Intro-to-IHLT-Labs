from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
import warnings
import pandas as pd

class ModelTrainer:
    """
    A class encapsulating various training methods for different models:
    Neural Network, MLP, Random Forest, and SVR.
    """

    @staticmethod
    def pearson_scorer(y_true, y_pred):
        # Returns a tuple (correlation, p-value)
        return pearsonr(y_true, y_pred)[0]

    def train_NN(self, df, input, output):
        """
        Train a neural network with grid search to find the best hyperparameters.
        """
        def create_model(input_dim, learning_rate = 0.001, neurons = 10, hidden_layers = 2):
            model = Sequential()
            model.add(Dense(neurons, input_dim=input_dim, activation='relu'))  # Input layer
            for _ in range(hidden_layers):  # Add hidden layers
                model.add(Dense(neurons, activation='relu'))
            model.add(Dense(1))  # Output layer
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
            return model

        X = df[input]
        y = df[output]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        input_dim = X.shape[1]

        model = KerasRegressor(
            model=create_model,
            model__input_dim=input_dim,  
            verbose=0
        )

        pearson_score = make_scorer(self.pearson_scorer, greater_is_better=True)

        param_grid = {
            "model__neurons": [5, 10],
            "model__hidden_layers": [2],
            "model__learning_rate": [0.001],
            "batch_size": [16, 32],
            "epochs": [50, 100],
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=pearson_score,
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)

        return best_model

    def train_MLP(self, df, input, output):
        """
        Train an MLP regressor using GridSearchCV to find the best hyperparameters.
        """
        X = df[input]
        y = df[output]

        # Initialize the MLPRegressor model
        model = MLPRegressor(max_iter=1000, early_stopping=True, validation_fraction=0.1, verbose=False)

        # Define custom scoring function (Pearson correlation)
        pearson_score = make_scorer(self.pearson_scorer, greater_is_better=True)

        # Hyperparameter grid
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,), (50, 100)],  
            'activation': ['relu', 'tanh'],                                       
            'solver': ['adam'],                                            
            'alpha': [0.0001, 0.001, 0.01],                                       
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [200, 500, 1000]                                                     
        }

        # GridSearchCV with error_score to handle failures
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=pearson_score,  
            cv=5,                             
            verbose=2,                        
            n_jobs=-1,                        
            error_score='raise'
        )

        # Suppress ConvergenceWarning during training
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            try:
                grid_search.fit(X, y)
            except ValueError as e:
                print("Error during GridSearchCV:", e)
                return None

        # Handle results
        results = grid_search.cv_results_
        failed_combinations = []
        for mean_score, params in zip(results['mean_test_score'], results['params']):
            if np.isnan(mean_score):
                failed_combinations.append(params)

        # Log failed combinations
        if failed_combinations:
            print(f"Failed combinations: {len(failed_combinations)}")
            for params in failed_combinations:
                print("Failed combination:", params)

        # Best model and hyperparameters
        best_model = grid_search.best_estimator_
        print("Best Hyperparameters:", grid_search.best_params_)

        return best_model

    def train_RF(self, df, input, output):
        """
        Train a Random Forest regressor using GridSearchCV to find the best hyperparameters.
        """
        X = df[input]
        y = df[output]

        model = RandomForestRegressor()

        param_grid = {
            'n_estimators': [100, 200, 300],  
            'max_depth': [None, 10], 
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],  
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],     
        }

        pearson_score = make_scorer(self.pearson_scorer, greater_is_better=True)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=pearson_score,  
            cv=5,                             
            verbose=2,                        
            n_jobs=-1                          
        )

        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        print("Best Hyperparameters:", grid_search.best_params_)

        # Plot feature importance
        feature_importances = pd.DataFrame({
            'Feature': input,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        N = 10  # Number of top features to display
        top_features = feature_importances.head(N)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features['Feature'], top_features['Importance'], edgecolor='k')
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {N} Important Features")
        ax.invert_yaxis()  # Invert y-axis for better readability

        return best_model, grid_search.best_params_, fig

    def train_single_RF(self, df, input, output, params):
        X = df[input]
        y = df[output]

        # Define the model
        pearson_score = make_scorer(self.pearson_scorer, greater_is_better=True)
        model = RandomForestRegressor(**params) #, scoring=pearson_score)

        # Fit the grid search to the data
        model.fit(X, y)

        return model

    def train_SVR(self, df, input, output):
        """
        Train an SVR model using RandomizedSearchCV to find the best hyperparameters.
        """
        X = df[input]
        y = df[output]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pearson_score = make_scorer(self.pearson_scorer, greater_is_better=True)

        model = SVR(cache_size=2000)

        param_grid = {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto']
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring=pearson_score,
            verbose=1,
            n_jobs=10,
        )

        random_search.fit(X_scaled, y)
        best_params = random_search.best_params_
        print("Best params:", best_params)
        best_model = random_search.best_estimator_

        return best_model