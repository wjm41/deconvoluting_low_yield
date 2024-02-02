from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

def generate_fingerprints(smile):
    """
    Generates a binary molecular fingerprint from a SMILES string.

    Parameters:
    - smile (str): SMILES notation of the molecule.

    Returns:
    - array (numpy.ndarray): A binary vector representing the molecular fingerprint.
    """
    mol = Chem.MolFromSmiles(smile)  # Convert SMILES to RDKit molecule object
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # Generate fingerprint
    array = np.zeros((0,), dtype=np.int8)  # Initialize an empty numpy array
    DataStructs.ConvertToNumpyArray(fp, array)  # Convert RDKit fingerprint to numpy array
    return array


def fit_forest(X, y):
    """
    Fits a RandomForestRegressor model using GridSearchCV to find optimal hyperparameters.

    Parameters:
    - X (numpy.ndarray): Feature matrix where rows are samples and columns are features.
    - y (numpy.ndarray): Target values.

    Returns:
    - model (RandomForestRegressor): The best RandomForest model found by GridSearchCV.
    """
    params = {'n_estimators': [100, 1000, 10000], 'max_depth': [1, 2, 3], 'min_samples_split': [2, 4]}
    # Define the parameter grid for GridSearchCV
    search = GridSearchCV(RandomForestRegressor(), params, cv=5, verbose=1, n_jobs=-1)
    # Initialize GridSearchCV with RandomForestRegressor, parameter grid, and cross-validation settings
    model = search.fit(X, y).best_estimator_  # Fit the model and retrieve the best estimator
    return model

# Read the dataset containing SMILES notations and activities
df = pd.read_csv('../data/process/activities_and_yield.csv')
print('Generating fingerprints...')
# Generate fingerprints for each molecule in the dataset
df['fps'] = [generate_fingerprints(smi) for smi in df["smiles"]]
# Calculate inhibition as 100 minus the mean activity percentage
df['inhibition'] = 100 - df['Mean activity (%)']

# Initialize empty lists to hold training and test dataframes
df_train = []
df_test = []
# Initialize an empty dataframe to store predictions
df_preds = pd.DataFrame()
# Initialize LeaveOneOut cross-validator
loo = LeaveOneOut()

print('Performing leave-one-out validation...')
# Perform leave-one-out cross-validation
for train_ind, test_ind in tqdm(loo.split(df['fps']), total=len(df)):
    # Split data into training and test sets based on the indices provided by LOO
    X_train, X_test = df['fps'].iloc[train_ind], df['fps'].iloc[test_ind]
    y_train, y_test = df['inhibition'].iloc[train_ind], df['inhibition'].iloc[test_ind]

    # Stack individual fingerprint arrays for training and test sets
    X_train = np.vstack(X_train.values)
    X_test = np.vstack(X_test.values)

    # Fit the RandomForest model on the training set
    rf_model = fit_forest(X_train, y_train)
    # Predict inhibition activity for the test compound
    test_pred = rf_model.predict(X_test)
    # Create a dataframe for the test split with actual and predicted values
    df_split = pd.DataFrame({'name': df['Molecule Name'].iloc[test_ind],
                             'smiles': df['smiles'].iloc[test_ind],
                             'Measured Inhibition (%)': y_test,
                             'RF Predicted Inhibition (%)': test_pred})
    # Concatenate predictions for each test split into a single dataframe
    df_preds = pd.concat([df_preds, df_split])
# Save the aggregated predictions to a CSV file
df_preds.to_csv('../data/predictions/rf_predictions.csv', index=False)
