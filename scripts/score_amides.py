# %% [markdown]
# # Score enumerated amides
#
# 1st June - notebook for running random forest to score the enumerated amides and inspect the top/bottom predictions.

# %%
import logging
import molplotly
import plotly.express as px
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import rdChemReactions
from rdkit import Chem
from typing import Union
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


# %% [markdown]
# Let's import the data and regenerate the amides from the amines

# %%
data_dir = '../data/'

df_amide_activity = pd.read_csv(
    f'{data_dir}/corrected_data_with_gp_predictions.csv')

# %%


def ensure_is_mol_object(smi_or_mol: Union[str, Mol],) -> Chem.Mol:
    if isinstance(smi_or_mol, str):
        definitely_a_mol = Chem.MolFromSmiles(smi_or_mol)
    elif isinstance(smi_or_mol, Mol):
        definitely_a_mol = smi_or_mol
    else:
        raise ValueError(
            f'{smi_or_mol} is not an RDKit Mol object!')
    return definitely_a_mol


def peptide_coupling_rxn() -> rdChemReactions.ChemicalReaction:

    acid_smarts = '[*:1][C:2](=O)O'
    amine_smarts = '[NX3;!$(N*=*):3]([H:4])[*:5]'
    amide_smarts = '[*:1][C:2](=O)[N:3][*:5]'

    peptide_rxn_smarts = f'{acid_smarts}.{amine_smarts}>>{amide_smarts}'
    peptide_rxn = rdChemReactions.ReactionFromSmarts(
        peptide_rxn_smarts)

    return peptide_rxn


def run_peptide_coupling(acid: Union[str, Mol],
                         amine: Union[str, Mol],
                         peptide_rxn: rdChemReactions.ChemicalReaction = None) -> Chem.Mol:

    try:
        acid_mol = ensure_is_mol_object(acid)
        amine_mol = ensure_is_mol_object(amine)

        reactants = tuple([Chem.AddHs(acid_mol), Chem.AddHs(amine_mol)])

        if peptide_rxn is None:
            peptide_rxn = peptide_coupling_rxn()

        possible_products = peptide_rxn.RunReactants(reactants)
        if len(possible_products) == 0:
            logging.warning(
                f'No product produced')
            return None
        else:
            product = Chem.RemoveHs(possible_products[0][0])
            return Chem.MolToSmiles(product)
    except Exception as e:
        print(amine)
        return None


# %%

acid = 'Clc1cc2c([C@H](C(Nc(cnc3)c4c3cccc4)=O)CN(CC(=O)O)C2)cc1'
acid = Chem.MolFromSmiles(acid)

peptide_rxn = peptide_coupling_rxn()

for index, row in tqdm(df_amide_activity.iterrows(), total=len(df_amide_activity)):
    df_amide_activity.loc[index, 'amide'] = run_peptide_coupling(
        acid, row.amine, peptide_rxn)

df_amide_activity = df_amide_activity.dropna()

# %% [markdown]
# Now let's fit the model

# %%


def generate_fingerprints(smile: str) -> np.ndarray:
    try:
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
    except Exception as e:
        print(f'Exception {e} for {smile}')
    return array


def fit_forest(X, y):
    params = {'n_estimators': [100, 1000, 10000], 'max_depth': [
        1, 2, 3], 'min_samples_split': [2, 4]}
    # params = {'n_estimators': [1000], 'max_depth': [1, 2]}
    # search = GridSearchCV(RandomForestRegressor(), params, cv=5, verbose=1, n_jobs=-1)
    # search = RandomizedSearchCV(
    #     RandomForestRegressor(), params, n_iter=10, n_jobs=-1, cv=3, scoring='neg_mean_squared_error', verbose=1)
    # model  = search.fit(X, y).best_estimator_
    # return model
    model = RandomForestRegressor(n_estimators=500)
    return model.fit(X, y)


# %%
df_amide_activity['fps'] = df_amide_activity['amide'].apply(
    generate_fingerprints)

X_train, y_train = df_amide_activity['fps'], df_amide_activity['inhibition']

X_train = np.vstack(X_train.values)

rf_model = fit_forest(X_train, y_train)

# %% [markdown]
# And lets run the model on the amide library

# %%
df_amide_library = pd.read_csv(
    f'{data_dir}/enumerated_amides.csv').dropna().drop_duplicates('amide')

df_amide_library['fps'] = df_amide_library['amide'].apply(
    generate_fingerprints)

X_test = np.vstack(df_amide_library.fps.values)

df_amide_library['rf'] = rf_model.predict(X_test)
df_amide_library

# %% [markdown]
# Lets load emmas data and compare the two

# %%
df_emma = pd.read_csv(f'{data_dir}/enamine_primary_and_secondary_amine_preds.csv',
                      usecols=['amine', 'pred_inhibition', 'std']).rename(columns={'amine': 'SMILES'})
df_merged = pd.merge(df_amide_library.drop(
    columns='fps'), df_emma, on='SMILES')
df_merged = df_merged.rename(
    columns={'rf': 'Random Forest', 'pred_inhibition': 'Gaussian Process'})
df_merged = df_merged.query(
    'amide not in @df_amide_activity.amide and std > 1e-3')
df_merged['mean_prediction'] = df_merged[[
    'Random Forest', 'Gaussian Process']].mean(axis=1)
df_merged = df_merged.round(3).sort_values('mean_prediction', ascending=False)
df_merged.to_csv(
    '/home/wjm41/ml_physics/noisyamides/data/predictions/enamine_primary_and_secondary_amine_both_preds.csv', index=False)
# %%

fig = px.scatter(df_merged, x='Random Forest', y='Gaussian Process', color='std', title="Correlation between Emma and William\'s models",
                 width=1400, height=1000)
app = molplotly.add_molecules(
    fig=fig, df=df_merged, title_col='ID', smiles_col=['SMILES', 'amide'], color_col='std', caption_cols=['std'], width=250, fontsize=16)

app.run_server(port=8060)
