# House Price Prediction using Linear Regression

This project involves predicting house prices based on their square footage, the number of bedrooms, and the number of bathrooms using a linear regression model.

## Project Overview

We use a linear regression model to predict the sale prices of houses. The model is trained on a dataset containing the following features:
- `GrLivArea`: Above grade (ground) living area square footage.
- `BedroomAbvGr`: Number of bedrooms above ground.
- `HalfBath`: Number of half bathrooms.
- `FullBath`: Number of full bathrooms.

The model is then used to predict house prices on a test dataset.

## Dataset

- `train.csv`: Training dataset containing features and target variable (`SalePrice`).
- `test.csv`: Test dataset containing features.
- `predicted_prices.csv`: Output file containing the test dataset with predicted house prices.

## Dependencies

- Python 3.x
- pandas
- scikit-learn

## Explanation

### Loading Data

The script starts by loading the training and test datasets using `pd.read_csv()`.

### Feature Extraction

- It extracts the relevant features (`GrLivArea`, `BedroomAbvGr`, `HalfBath`, `FullBath`) from the training dataset to create `X_train`.
- The target variable (`SalePrice`) is extracted to create `y_train`.
- The same features are extracted from the test dataset to create `X_test`.

### Model Training

A linear regression model is instantiated and trained using `model.fit(X_train, y_train)`.

### Prediction

The model predicts house prices for the test dataset using `model.predict(X_test)`.

### Saving Results

- The rounded predicted prices are added as a new column to the test dataset.
- The test dataset with the predicted prices is saved to `predicted_prices.csv`.

## Results

The results are saved in `predicted_prices.csv`, containing the `Id` of each house and its corresponding `PredictedPrice`.

## Acknowledgments

The data used in this project is provided by Kaggle.
