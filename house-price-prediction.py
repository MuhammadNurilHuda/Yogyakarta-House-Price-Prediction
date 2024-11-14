'''
!!!

This code is implemented only with Random Forest, as it provides the best results.

!!!
'''

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import category_encoders as ce

# Function to load data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset and perform initial cleaning.
    
    Parameters:
        file_path (str): Path to the dataset.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = pd.read_csv(file_path)
    df.drop(columns=['nav-link', 'description'], inplace=True)
    df.dropna(subset=['bed', 'bath', 'surface_area', 'building_area'], inplace=True)
    df['carport'] = df['carport'].fillna(0)
    df['listing-location'] = [re.sub(r'\s+', '', i).strip() for i in df['listing-location']]
    df.drop_duplicates(inplace=True)
    return df

# Function to preprocess 'price'
def convert_price(price_str: str) -> int:
    """
    Convert price from string to numeric format.
    
    Parameters:
        price_str (str): Price as string.
        
    Returns:
        int: Price in integer format.
    """
    price_str = price_str.replace("Rp ", "")
    if "Miliar" in price_str:
        number = float(re.sub("[^0-9,]", "", price_str).replace(",", ".")) * 1_000_000_000
    elif "Juta" in price_str:
        number = float(re.sub("[^0-9,]", "", price_str).replace(",", ".")) * 1_000_000
    else:
        number = float(re.sub("[^0-9,]", "", price_str).replace(",", "."))
    return int(number)

# Function to convert a column to integer
def convert_to_int(data: pd.Series) -> pd.Series:
    """
    Convert a column of data to integer type.
    
    Parameters:
        data (pd.Series): Data column to convert.
        
    Returns:
        pd.Series: Converted data in integer type.
    """
    return data.astype(int)

# Function to preprocess 'area'
def convert_area(area_str: str) -> int:
    """
    Convert area from string to integer format.
    
    Parameters:
        area_str (str): Area as string.
        
    Returns:
        int: Area in integer format.
    """
    return int(area_str.replace("m²", "").strip())

# Function to apply all preprocessing
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to the dataframe.
    
    Parameters:
        df (pd.DataFrame): Dataframe to preprocess.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    df['price'] = df['price'].apply(convert_price)
    df[['bed', 'bath', 'carport']] = df[['bed', 'bath', 'carport']].apply(convert_to_int)
    df['surface_area'] = df['surface_area'].apply(convert_area)
    df['building_area'] = df['building_area'].apply(convert_area)
    
    print("\n")
    print("=========================================================================")    
    print("Data has been preprocessed....")
    print(df.head(2))
    print("=========================================================================")    

    return df

# Function to encode categorical features
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using target encoding.
    
    Parameters:
        df (pd.DataFrame): Dataframe to encode.
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features.
    """
    encoders = ce.TargetEncoder(cols=['listing-location'])
    df['listing-location'] = encoders.fit_transform(df['listing-location'], df['price'])
    
    print("\n")
    print("=========================================================================")    
    print("Data has been encoded....")
    print(df.head(2))
    print("=========================================================================")    
    
    return df

# Function to normalize data
def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data using Standard Scaler.
    
    Parameters:
        df (pd.DataFrame): Dataframe to normalize.
        
    Returns:
        pd.DataFrame: Normalized dataframe.
    """
    scaler = StandardScaler()
    df.drop(columns='building_area', inplace=True) # Drop 'building_area' due low of correlation
    scaled = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    
    print("\n")
    print("=========================================================================")    
    print("Data has been normalized....")
    print(normalized_df.head(2), df.shape)
    print("=========================================================================")    

    return normalized_df

# Function to train a model and evaluate it
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    """
    Train a model and evaluate using Mean Squared Error.
    
    Parameters:
        X_train, X_test, y_train, y_test: Training and test sets.
        model: Machine learning model to train.
        
    Returns:
        None
    """
    model.fit(X_train, y_train)
    
    # Evaluate on train set
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    print(f"Train Mean Squared Error: {train_mse}")
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Test Mean Squared Error: {test_mse}")

# Main function to run the full pipeline
def main(file_path: str):
    # Load and preprocess data
    df = load_data(file_path)
    df = preprocess_data(df)
    df = encode_features(df)

    # Normalize the entire dataset first
    df = normalize_data(df)

    # Separate features and target variable
    X = df.drop(columns=['price'])
    y = df['price']

    # Split normalized data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Continue with model training and evaluation
    rf_model = RandomForestRegressor(
        bootstrap=False, max_depth=None, max_features='sqrt', min_samples_leaf=1,
        min_samples_split=2, n_estimators=50, n_jobs=-1, oob_score=False, random_state=42
    )
    print("\n")
    print("=========================================================================")
    print("Random Forest Results:")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, rf_model)
    print("=========================================================================")    

# Run the main function
if __name__ == "__main__":
    main("Dataset/rumah123_yogya_unfiltered.csv")