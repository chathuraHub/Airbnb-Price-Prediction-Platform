# Airbnb Price Prediction Platform

An end-to-end machine learning project for predicting Airbnb listing prices using property characteristics, location information, amenities, and host-related features. The project combines exploratory data analysis, preprocessing, regression model training, feature engineering, model evaluation, and a Flask-based web application for real-time price prediction.

## Features

- Interactive web interface for Airbnb price prediction
- Data preprocessing, missing-value handling, and feature scaling
- Ordinal and One-Hot Encoding experiments
- Comparison of multiple regression algorithms
- CatBoost modeling with location-based feature engineering
- Model evaluation using R², RMSE, and MAE
- SHAP-based feature importance and model explainability
- Modular machine learning pipelines for training and inference

## Tech Stack

**Programming:** Python

**Data Analysis:** Pandas, NumPy

**Machine Learning:** Scikit-learn, CatBoost, XGBoost

**Visualization and Explainability:** Matplotlib, Seaborn, SHAP

**Web Application:** Flask, HTML, CSS

**Tools:** Git, GitHub, Jupyter Notebook

## Project Workflow

The machine learning workflow consists of the following stages:

1. **Exploratory Data Analysis:** Examine feature distributions, missing values, and relationships between property characteristics and listing prices.
2. **Data Ingestion:** Load the Airbnb dataset and split labeled data into training and validation sets.
3. **Data Transformation:** Process numerical and categorical features, handle missing values, scale numerical inputs, and apply categorical encoding.
4. **Model Training:** Train and compare regression algorithms, including Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, and CatBoost.
5. **Feature Engineering:** Experiment with geographic features such as location grid cells, K-Means location clusters, and distances.
6. **Model Evaluation:** Evaluate predictions using R², RMSE, and MAE.
7. **Model Explainability:** Generate SHAP visualizations to understand feature contributions.
8. **Web Deployment:** Serve the saved prediction model through an interactive Flask application.

The preprocessing and prediction pipelines use logarithmically transformed listing prices and convert predictions back to the original price scale using exponentiation.

## Project Structure

```text
Airbnb-Price-Prediction-Platform/
│
├── app.py
├── requirements.txt
├── setup.py
├── Dockerfile
│
├── src/
│   └── Airbnb/
│       ├── components/
│       │   ├── Data_ingestion.py
│       │   ├── Data_transformation.py
│       │   └── Model_trainer.py
│       │
│       ├── pipelines/
│       │   ├── Training_pipeline.py
│       │   ├── Prediction_Pipeline.py
│       │   └── CatBoost_pipeline.py
│       │
│       ├── utils/
│       │   └── utils.py
│       │
│       ├── exception.py
│       └── logger.py
│
├── airbnb-price-prediction/
│   ├── Airbnb_train.csv
│   ├── Airbnb_test.csv
│   ├── Model_ordinal.pkl
│   ├── Model_onehot.pkl
│   ├── Preprocessor_ordinal.pkl
│   ├── Preprocessor_onehot.pkl
│   └── CatBoost_*.cbm
│
├── Notebook_Experiments/
│   ├── Airbnb_Price_Prediction.ipynb
│   └── Exploratory_Data_Analysis.ipynb
│
├── templates/
│   ├── index.html
│   ├── error.html
│   └── _macros.html
│
├── static/
│   ├── style.css
│   └── images/
│
├── reports/
│   ├── shap_bar.png
│   └── shap_beeswarm.png
│
├── check_model.py
├── explain_model.py
└── generate_submission.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/chathuraHub/Airbnb-Price-Prediction-Platform.git
cd Airbnb-Price-Prediction-Platform
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For SHAP visualizations, also install:

```bash
pip install shap matplotlib
```

### 4. Run the application

```bash
python app.py
```

Open the application in your browser:

**http://localhost:8080**

Enter the property details and submit the form to receive the predicted listing price.

The application uses the saved ordinal-encoded preprocessing pipeline and regression model by default.

## Model Training

The repository includes saved model files, so retraining is not required to use the web application.

To reproduce the ordinal and one-hot encoding experiments:

```bash
python -m src.Airbnb.pipelines.Training_pipeline
```

This pipeline performs data ingestion, preprocessing, model comparison, training, and serialization.

To train the separate CatBoost model with geographic grid-cell features:

```bash
python -m src.Airbnb.pipelines.CatBoost_pipeline
```

This creates `CatBoost_model.cbm` and `CatBoost_model_metrics.json` in the `airbnb-price-prediction` directory.

The CatBoost model is maintained separately from the default Flask prediction pipeline.

## Model Evaluation

To evaluate the saved ordinal-encoded and one-hot-encoded regression models:

```bash
python check_model.py
```

Evaluation includes:

- R² on log-transformed prices
- RMSE on log-transformed prices
- R² on original listing prices
- MAE on original listing prices

### CatBoost Location Experiments

Additional saved CatBoost experiments explore different approaches to representing geographic information:

| Experiment | Location Representation | Test R² | Test MAE |
|---|---|---:|---:|
| Native CatBoost | Latitude and longitude | 0.5720 | $53.22 |
| Grid-cell experiment | Rounded geographic coordinates | 0.5748 | $53.00 |
| K-Means experiment | Location clusters | 0.5673 | $53.72 |
| Distance experiment | Distance-based feature | 0.5728 | $53.23 |

These results are taken from the saved experiment metrics, evaluated with the original price as the target. The grid-cell experiment achieved the highest test R² among these four saved experiments.

## Model Explainability

To generate SHAP-based visualizations for the saved regression model:

```bash
python explain_model.py
```

The script produces:

- `reports/shap_bar.png` — Global feature importance
- `reports/shap_beeswarm.png` — Feature contributions and their directions

These visualizations help interpret how different listing characteristics affect predicted prices.

## Batch Prediction

The repository also contains a batch submission script:

```bash
python generate_submission.py
```

This script uses the legacy default model files, `Model.pkl` and `Preprocessor.pkl`, to predict prices for records in `Airbnb_test.csv` and save a submission CSV.

## Web Application

The Flask interface allows users to enter listing information, including:

- Property and room type
- Amenities and accommodation capacity
- Bathrooms, bedrooms, and beds
- City, latitude, and longitude
- Cancellation policy and cleaning fee
- Host verification and booking information
- Number of reviews and review rating

The application preprocesses these inputs, performs model inference, and displays an estimated listing price.

## Future Improvements

- Integrate the independently trained CatBoost model into the Flask interface
- Add model selection and comparison in the web application
- Display feature-level prediction explanations
- Improve geographic feature engineering
- Add automated tests and deployment workflows

## Author

**Chathura**

GitHub: https://github.com/chathuraHub

## Disclaimer

This project is intended for machine learning experimentation and educational purposes. Predicted listing prices are estimates and may differ from actual Airbnb prices.
