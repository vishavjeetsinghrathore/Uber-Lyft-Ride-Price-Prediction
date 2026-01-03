# train_model.py

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from evaluate_model import evaluate_model


# ==================================================
# 1. Load Dataset
# ==================================================
df = pd.read_csv("data/cab_rides.csv")

# Drop unnecessary columns
df = df.drop(columns=["id", "product_id", "name"], errors="ignore")

# ===============================
# IMPORTANT FIX: Remove NaN target
# ===============================
df = df.dropna(subset=["price"])
df.reset_index(drop=True, inplace=True)

print("Dataset shape after cleaning:", df.shape)


# ==================================================
# 2. Split Features & Target
# ==================================================
X = df.drop("price", axis=1)
y = df["price"]

print("NaN values in target:", y.isna().sum())


# ==================================================
# 3. Feature Types
# ==================================================
categorical_features = ["cab_type", "destination", "source"]
numerical_features = ["distance", "surge_multiplier"]


# ==================================================
# 4. Preprocessing
# ==================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# ==================================================
# 5. Train-Test Split
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================================================
# 6. Models to Compare
# ==================================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}


# ==================================================
# 7. Model Comparison with Cross-Validation
# ==================================================
results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    # Train & Predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse, rmse, mae, r2, adj_r2 = evaluate_model(
        y_test, y_pred, X_train
    )

    results.append([
        name,
        np.mean(cv_scores),
        mse,
        rmse,
        mae,
        r2,
        adj_r2
    ])


results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "CV_R2",
        "MSE",
        "RMSE",
        "MAE",
        "R2",
        "Adjusted R2"
    ]
)

print("\nModel Comparison with Cross-Validation:\n")
print(results_df)


# ==================================================
# 8. Select Best Model (Highest CV_R2)
# ==================================================
best_model_name = results_df.sort_values(
    "CV_R2", ascending=False
).iloc[0]["Model"]

print(f"\nBest Model Selected: {best_model_name}")

best_base_model = models[best_model_name]


# ==================================================
# 9. Hyperparameter Tuning (ONLY for Selected Model)
# ==================================================
param_grids = {
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5]
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    },
    "Decision Tree": {
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }
}

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_base_model)
])


if best_model_name in param_grids:
    print("\nApplying Hyperparameter Tuning...")

    grid_search = GridSearchCV(
        final_pipeline,
        param_grids[best_model_name],
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    final_model = grid_search.best_estimator_

    print("Best Hyperparameters:")
    print(grid_search.best_params_)

else:
    print("\nNo hyperparameter tuning required for this model.")
    final_pipeline.fit(X_train, y_train)
    final_model = final_pipeline


# ==================================================
# 10. Save Final Model
# ==================================================
joblib.dump(final_model, "best_model.pkl")
print("\nFinal model saved as best_model.pkl")
