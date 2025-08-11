import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# ============================
# FEA Batch Model
# ============================
class OneD:
    def __init__(self, Elements, Nodes, k): 
        self.Elements = Elements
        self.Nodes = Nodes
        self.K_val = k
        self.K_matrix = np.zeros((self.Nodes, self.Nodes))
        self.Local_k = np.array([[1, -1], [-1, 1]])
        self.Force = np.zeros((self.Nodes, 1))

    def K_stiffness(self):
        for j in range(0, self.Nodes - 1):
            self.K_matrix[j:j+2, j:j+2] += self.K_val * self.Local_k
        return self.K_matrix

    def apply_boundary_condition(self, BC, Location, Quantity):
        if BC == 0:  # Essential (Dirichlet)
            self.Force[Location] += self.K_matrix[Location, Location] * Quantity
            self.K_matrix = np.delete(self.K_matrix, Location, axis=0)
            self.K_matrix = np.delete(self.K_matrix, Location, axis=1)
            self.Force = np.delete(self.Force, Location, axis=0)

        elif BC == 1:  # Natural (Neumann)
            self.Force[Location] += Quantity

        elif BC == 2:  # Mixed
            self.K_matrix[Location, Location] += self.K_val
            self.Force[Location] += self.K_val * Quantity

    def solve(self):
        try:
            T = np.linalg.solve(self.K_matrix, self.Force)
            return T
        except np.linalg.LinAlgError:
            return None

# Exhaustive generation parameters
Nodes = 10
Elements = Nodes - 1
k_values = np.linspace(5, 200, 100)  # Vary stiffness from 5 to 200
BC_types = [0, 1, 2]
Quantity_range = np.linspace(1, 150, 200)  #  values from 1 to 150

# Store results
dataset = []

# Loop through k values, all nodes, BC types and quantities
for k in k_values:
    for node in range(Nodes):
        for BC_type in BC_types:
            for quantity in Quantity_range:
                model = OneD(Elements, Nodes, k)
                model.K_stiffness()
                model.apply_boundary_condition(BC_type, node, quantity)
                solution = model.solve()
                if solution is not None:
                    # Flatten and zero-pad result if needed
                    temp_vec = solution.flatten()
                    padded_vec = np.pad(temp_vec, (0, Nodes - len(temp_vec)), 'constant')
                    dataset.append(np.append([k, BC_type, node, quantity], padded_vec))

# Convert to numpy array
dataset = np.array(dataset)

# Assemble final features with stiffness matrix
final_features = []
for row in dataset:
    k_val, bc_type, node, quantity = row[:4]
    temps = row[4:]

    model = OneD(Elements, Nodes, k_val)
    K_matrix = model.K_stiffness()
    flat_K = K_matrix.flatten()
    nonzero_flat_K = flat_K[flat_K != 0]

    feature_row = np.concatenate((temps, [k_val, bc_type, node, quantity], nonzero_flat_K))
    final_features.append(feature_row)

final_features = np.array(final_features)

# ============================
# Machine Learning Model
# ============================

# Prepare ML data
X = final_features[:, 10:]  # All features except T1 to T10
y = final_features[:, :10]  # T1 to T10 as regressors

# Scale target values (T1 to T10)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y)

# Split the dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline with scaler and MultiOutputRegressor using GradientBoosting
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("gb", MultiOutputRegressor(GradientBoostingRegressor()))
])

# Define hyperparameter grid
param_grid = {
    "gb__estimator__n_estimators": [100],
    "gb__estimator__learning_rate": [0.01],
    "gb__estimator__max_depth": [5],
    "gb__estimator__min_samples_split": [2],
    "gb__estimator__min_samples_leaf": [1],
    "gb__estimator__subsample": [0.8]
}

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)

# Fit model on training data
grid_search.fit(X_train, y_train)

# After predicting, inverse the scaling on the predictions
y_train_pred = grid_search.predict(X_train)
y_test_pred = grid_search.predict(X_test)

y_train_pred_inv = y_scaler.inverse_transform(y_train_pred)
y_test_pred_inv = y_scaler.inverse_transform(y_test_pred)

# Evaluate on training data
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Evaluate on test data
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Best model and metrics
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (neg MSE):", -grid_search.best_score_)
print(f"Training MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}")
