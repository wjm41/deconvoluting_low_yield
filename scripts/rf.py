# %% [markdown]
# ### Imports

# %%
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

import molplotly


# %% [markdown]
# ### Define Useful functions

# %%
def generate_fingerprints(smile):
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) 
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
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


def calc_metrics(model, model_name, X_train, y_train, X_test, y_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    r2_train = r2_score(y_train, train_preds)
    rho_train = spearmanr(y_train, train_preds)[0]
    rmse_train = np.sqrt(mean_squared_error(y_train, train_preds))

    r2_test = r2_score(y_test, test_preds)
    rho_test = spearmanr(y_test, test_preds)[0]
    rmse_test = np.sqrt(mean_squared_error(
        y_test, test_preds))

    df_train = pd.DataFrame({'model': model_name,
                             'R2': r2_train,
                             'rho': rho_train,
                             'RMSE': rmse_train},
                            index=[0])
    df_test = pd.DataFrame({'model': model_name,
                            'R2': r2_test,
                            'rho': rho_test,
                            'RMSE': rmse_test},
                           index=[0])

    return test_preds, df_train, df_test


def calc_metrics_simple(model_name, y_test, test_preds):

    r2_test = r2_score(y_test, test_preds)
    rho_test = spearmanr(y_test, test_preds)[0]
    rmse_test = np.sqrt(mean_squared_error(
        y_test, test_preds))

    df_test = pd.DataFrame({'model': model_name,
                            'R2': r2_test,
                            'rho': rho_test,
                            'RMSE': rmse_test},
                           index=[0])

    return df_test


# %% [markdown]
# ### Load data

# %%
# if not os.path.isfile('loo_predictions.csv'):
data_dir = '../data/'
df = pd.read_csv(data_dir+'noisy_amides.csv')
df = df.drop_duplicates(subset='SMILES')
df.to_csv('noisy_amides.csv', index=False)
# pharmacophore fingerprint
df['fps'] = [generate_fingerprints(smi) for smi in df["SMILES"]]
# df['fps'] = np.vstack(df['fps'].values)
df['inhibition'] = 100 - df['Mean activity (%)']
print('Number of non-duplicate entries: {}'.format(len(df)))
print('Shape of Fingerprint: {}'.format(df['fps'].iloc[0].shape))


# %% [markdown]
# ### Perform Leave-One-Out training with random forest

# %%
df_train = []
df_test = []
df_preds = pd.DataFrame()
loo = LeaveOneOut()
i = 0
test = False
for train_ind, test_ind in tqdm(loo.split(df['fps']), total=len(df)):
    X_train, X_test = df['fps'].iloc[train_ind], df['fps'].iloc[test_ind]
    y_train, y_test = df['inhibition'].iloc[train_ind], df['inhibition'].iloc[test_ind]

    X_train = np.vstack(X_train.values)
    X_test = np.vstack(X_test.values)

    rf_model = fit_forest(X_train, y_train)
    test_pred = rf_model.predict(X_test)
    df_split = pd.DataFrame({'name': df['Molecule Name'].iloc[test_ind],
                            'SMILES': df['SMILES'].iloc[test_ind],
                                'Measured Inhibition (%)': y_test,
                                'RF Predicted Inhibition (%)': test_pred})
    df_preds = pd.concat([df_preds, df_split])
    i += 1
    if test:
        if i > 10:
            break
df_preds.to_csv('loo_predictions.csv', index=False)
df_test = calc_metrics_simple(
    'Naive RF', df['inhibition'].iloc[:i], df_preds['RF Predicted Inhibition (%)'])

print('\nTest Set')
print(df_test.to_string(index=False))

# else:
#     df_preds = pd.read_csv('loo_predictions.csv')

df_metrics = calc_metrics_simple(
        'Naive RF', df_preds['Measured Inhibition (%)'], df_preds['RF Predicted Inhibition (%)']).iloc[0]
df_metrics

# %% [markdown]
# #### Scatter Plot of LOO results

# %%
df_preds = pd.read_csv('../data/predictions/loo_predictions.csv')
df_metrics = calc_metrics_simple(
    'Naive RF', df_preds['Measured Inhibition (%)'], df_preds['RF Predicted Inhibition (%)']).iloc[0]
df_metrics


# %%
import time
df_preds = pd.read_csv('../data/predictions/loo_predictions.csv')
df_metrics = calc_metrics_simple(
    'Naive RF', df_preds['Measured Inhibition (%)'], df_preds['RF Predicted Inhibition (%)']).iloc[0]
fig = go.Figure(data=[
    go.Scatter(
        x=df_preds["Measured Inhibition (%)"],
        y=df_preds["RF Predicted Inhibition (%)"],
        marker=dict(
        # color=np.abs(df_preds["RF Predicted Inhibition (%)"] - df_preds["Measured Inhibition (%)"]),
        # colorscale='RdBu',
        # colorscale='Blues',
        # cmid=0,
        # cmax=10,
        # cmin=-10,
        # colorbar=dict(title='Δ Inhibition'),
        # showscale=True
        ),
        mode="markers",
        # template='simple_white'
    )
])


fig.update_layout(
    title='Leave-One-Out Regression',
    # title=r'$Leave-One-Out Regression (N=300, R^2: , \rho:)$',
    autosize=True,
    width=1200, # size of figure
    height=800,
    xaxis=dict(title="Measured Inhibition (%)"),
    yaxis=dict(title="RF Predicted Inhibition (%)"),
    template='plotly_white'
)

fig.add_shape(type='line',
              x0=0,
              y0=0,
              x1=100,
              y1=100,
              line=dict(color='black', dash='dash'),
              xref='x',
              yref='y'
              )

fig.write_image('../results/rf.pdf')
time.sleep(1)
fig.write_image('../results/rf.pdf')
fig.show()


