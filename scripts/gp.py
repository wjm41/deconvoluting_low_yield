from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.model_selection import LeaveOneOut, GridSearchCV
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

def remove_acid(smiles):
    """
    Removing the hardcoded acid fragment from the input smiles to return just the amine.
    
    Parameters:
    - smiles (str): SMILES notation of the molecule.
    """
    # The acid fragment smiles.
    acid = 'Clc1cc2c([C@H](C(Nc(cnc3)c4c3cccc4)=O)CN(CC=O)C2)cc1'
    acid = Chem.MolFromSmarts(acid)

    peptide = Chem.MolFromSmiles(smiles)

    # Removing acid fragment from the peptide.
    amine = AllChem.DeleteSubstructs(peptide, acid)
    amine.UpdatePropertyCache()
    Chem.GetSymmSSSR(amine)

    return amine


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


def fit_and_predict_gp(X_train, y_train, X_test, kernel_type='RBF'):
    """
    Fits a GaussianProcessRegressor model with the specified kernel and predicts the test set.

    Parameters:
    - X_train (numpy.ndarray): Training feature matrix.
    - y_train (numpy.ndarray): Training target vector.
    - X_test (numpy.ndarray): Test feature matrix.
    - kernel_type (str): Type of kernel to use ('RBF' or 'Matern').

    Returns:
    - test_pred (numpy.ndarray): Predicted values for the test set.
    """
    if kernel_type == 'RBF':
        params = {"kernel": [RBF(l) for l in np.logspace(-1, 1, 3)]}

    elif kernel_type == 'Matern':
        params = {"kernel": [Matern(l) for l in np.logspace(-1, 1, 3)]}

    else:
        raise ValueError("Unsupported kernel type. Use 'RBF' or 'Matern'.")

    # Define the parameter grid for GridSearchCV
    search = GridSearchCV(GaussianProcessRegressor(), params, cv=5, verbose=1, n_jobs=-1)
    # Initialize GridSearchCV with RandomForestRegressor, parameter grid, and cross-validation settings
    gp_model = search.fit(X_train, y_train).best_estimator_  # Fit the model and retrieve the best estimator
    test_pred = gp_model.predict(X_test)  # Predict on the test set
    return test_pred

# Read the dataset containing SMILES notations and activities
df = pd.read_csv('../data/process/activities_and_yield.csv')
print('Generating fingerprints...')
# Generate fingerprints for each molecule in the dataset
df['amine'] = [remove_acid(smi) for smi in df["smiles"]]
df['fps'] = [generate_fingerprints(smi) for smi in df["amine"]]
# Calculate inhibition as 100 minus the mean activity percentage
df['inhibition'] = 100 - df['Mean activity (%)']

# Initialize an empty dataframe to store predictions
df_preds = pd.DataFrame()
# Initialize LeaveOneOut cross-validator
loo = LeaveOneOut()

print('Performing leave-one-out validation with Gaussian Process models...')
# Perform leave-one-out cross-validation using Gaussian Process models
for train_ind, test_ind in tqdm(loo.split(df['fps']), total=len(df)):
    # Split data into training and test sets based on the indices provided by LOO
    X_train, X_test = df['fps'].iloc[train_ind], df['fps'].iloc[test_ind]
    y_train, y_test = df['inhibition'].iloc[train_ind], df['inhibition'].iloc[test_ind]

    # Stack individual fingerprint arrays for training and test sets
    X_train = np.vstack(X_train.values)
    X_test = np.vstack(X_test.values)

    # Fit the Gaussian Process model with RBF kernel and predict on the test set
    test_pred_rbf = fit_and_predict_gp(X_train, y_train, X_test, kernel_type='RBF')
    # Fit the Gaussian Process model with Matern kernel and predict on the test set
    test_pred_matern = fit_and_predict_gp(X_train, y_train, X_test, kernel_type='Matern')

    # Create a dataframe for the test split with actual and predicted values using both kernels
    df_split = pd.DataFrame({'name': df['Molecule Name'].iloc[test_ind],
                             'smiles': df['smiles'].iloc[test_ind],
                             'Measured Inhibition (%)': y_test,
                             'GP RBF Predicted Inhibition (%)': test_pred_rbf,
                             'GP Matern Predicted Inhibition (%)': test_pred_matern})
    # Concatenate predictions for each test split into a single dataframe
    df_preds = pd.concat([df_preds, df_split])

# Save the aggregated predictions to a CSV file
df_preds.to_csv('../data/predictions/gp_predictions.csv', index=False)
