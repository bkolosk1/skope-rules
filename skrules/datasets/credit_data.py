"""
Default of Credit Card Clients Dataset.

The original database is available from the UCI Machine Learning Repository:

    https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

The data contains 30,000 observations on 24 variables.

References
----------

Lichman, M. (2013). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer
Science.
"""

import pandas as pd
import numpy as np
import os
import urllib.request
from sklearn.datasets import get_data_home
from sklearn.utils import Bunch


def load_credit_data():
    """
    Load the Default of Credit Card Clients dataset.

    Returns
    -------
    dataset : sklearn.utils.Bunch
        Dictionary-like object with the following attributes:

        data : ndarray of shape (30000, 23)
            The data matrix excluding the target variable.

        target : ndarray of shape (30000,)
            The target vector indicating default payment next month.

        feature_names : list of str
            The names of each feature.
    """
    # Define the URL and filename
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
    filename = 'default_of_credit_card_clients.xls'
    
    # Get the scikit-learn data directory
    data_home = get_data_home()
    filepath = os.path.join(data_home, filename)
    
    # Download the file if it does not exist
    if not os.path.exists(filepath):
        print(f"Downloading '{filename}' from {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download the dataset from {url}.") from e
    else:
        print(f"File '{filename}' already exists in '{data_home}'. Skipping download.")
    
    # Read the Excel file
    try:
        data = pd.read_excel(filepath, sheet_name='Data', header=1)
    except Exception as e:
        raise RuntimeError(f"Failed to read the dataset from '{filepath}'.") from e
    
    # Rename 'PAY_0' to 'PAY_1' if it exists
    if 'PAY_0' in data.columns:
        data = data.rename(columns={"PAY_0": "PAY_1"})
    
    # Extract features and target
    X = data.drop('default payment next month', axis=1)
    y = data['default payment next month'].values
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Convert to numpy arrays
    X_values = X.values
    y_values = y
    
    # Create a Bunch object
    dataset = Bunch(
        data=X_values,
        target=y_values,
        feature_names=feature_names
    )
    
    return dataset
