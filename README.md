# Uber and Lyft Ride Prices Prediction

This project is a machine learning application for predicting ride prices for Uber and Lyft based on factors like distance, surge multiplier, cab type, source, and destination. It includes model training, evaluation, and a Streamlit web app for interactive price prediction.

## Features

- **Data Preprocessing**: Handles categorical and numerical features using OneHotEncoding and StandardScaler.
- **Model Training**: Compares multiple regression models (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting) with cross-validation.
- **Hyperparameter Tuning**: Optimizes selected models using GridSearchCV.
- **Model Evaluation**: Provides metrics like MSE, RMSE, MAE, R², and Adjusted R².
- **Web App**: Interactive Streamlit app for real-time price predictions.
- **Model Persistence**: Saves the best trained model for deployment.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Uber-and-Lyft-ride-prices-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the training script to train and select the best model:

```bash
python train_model.py
```

This will:
- Load and preprocess the data from `data/cab_rides.csv`.
- Compare models using cross-validation.
- Perform hyperparameter tuning on the best model.
- Save the final model as `best_model.pkl`.

### Running the Prediction App

Launch the Streamlit web app:

```bash
streamlit run app.py
```

Open the provided URL in your browser. Enter ride details (distance, surge multiplier, cab type, source, destination) to get a price prediction.

## Data

The dataset (`data/cab_rides.csv`) contains ride information including:
- `distance`: Distance of the ride in km.
- `surge_multiplier`: Surge pricing factor.
- `cab_type`: Type of cab (Uber or Lyft).
- `source`: Pickup location.
- `destination`: Drop-off location.
- `price`: Target variable (ride price).

Ensure the data file is placed in the `data/` directory.

## Model Details

- **Preprocessing**: Uses `ColumnTransformer` with `StandardScaler` for numerical features and `OneHotEncoder` for categorical features.
- **Models Evaluated**:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Selection**: Best model chosen based on cross-validation R² score.
- **Tuning**: Hyperparameter optimization for tree-based models using GridSearchCV.

## Project Structure

```
Uber-and-Lyft-ride-prices-prediction/
├── app.py                 # Streamlit web app for predictions
├── train_model.py         # Model training and evaluation script
├── evaluate_model.py      # Custom evaluation function
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/
│   └── cab_rides.csv      # Dataset
└── best_model.pkl         # Saved model (generated after training)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Project Live On

https://uber-lyft-ride-price-prediction-4xthqsfccvmikujvp3mj7r.streamlit.app/