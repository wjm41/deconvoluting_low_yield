from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

def generate_fingerprints(smile):
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) 
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def fit_forest(X, y):
    params = {'n_estimators': [100, 1000, 10000], 'max_depth': [
        1, 2, 3], 'min_samples_split': [2, 4]}
    search = GridSearchCV(RandomForestRegressor(), params, cv=5, verbose=1, n_jobs=-1)
    model = search.fit(X, y).best_estimator_
    return model

df = pd.read_csv('../data/process/activities_and_yield.csv')
print('Generating fingerprints...')
df['fps'] = [generate_fingerprints(smi) for smi in df["smiles"]]
df['inhibition'] = 100 - df['Mean activity (%)']


df_train = []
df_test = []
df_preds = pd.DataFrame()
loo = LeaveOneOut()

print('Performing leave-one-out validation...')
for train_ind, test_ind in tqdm(loo.split(df['fps']), total=len(df)):
    X_train, X_test = df['fps'].iloc[train_ind], df['fps'].iloc[test_ind]
    y_train, y_test = df['inhibition'].iloc[train_ind], df['inhibition'].iloc[test_ind]

    X_train = np.vstack(X_train.values)
    X_test = np.vstack(X_test.values)

    rf_model = fit_forest(X_train, y_train)
    test_pred = rf_model.predict(X_test)
    df_split = pd.DataFrame({'name': df['Molecule Name'].iloc[test_ind],
                            'smiles': df['smiles'].iloc[test_ind],
                                'Measured Inhibition (%)': y_test,
                                'RF Predicted Inhibition (%)': test_pred})
    df_preds = pd.concat([df_preds, df_split])
df_preds.to_csv('../data/predictions/rf_predictions.csv', index=False)