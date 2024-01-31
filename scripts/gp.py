'''
A Gaussian Process Regressor for predicting inhibition activity of various amide scaffolds.

THERE ARE 300 UNIQUE SMILES AND 300 UNIQUE CANONICAL SMILES

Leave one out.
'''
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

# Defining the kernels
rbf_kernel = RBF()
matern_kernel = Matern()

rbf_gp = GaussianProcessRegressor(kernel=rbf_kernel)
matern_gp = GaussianProcessRegressor(kernel=matern_kernel)

'''
Splitting the dataframe to training and testing datasets. Leave
out user defined percentage of molecules.
    Args:
        df (dataframe): The pre-split dataframe.
        
        test_ratio (float): The percentage of molecules
                            to be left out for testing.
'''
def train_test_split(df, test_ratio=0.05):
    random.seed(1)
    smiles = df['amine']
    # Finding the smiles that will be used for the test dataset.
    test_smiles = random.sample(smiles.to_list(), int(np.ceil(test_ratio*len(df))))
    test_df = pd.DataFrame()

    # Splitting the dataframe into test and non-test (train).
    for smile in test_smiles:
        test_row = df[df['amine'] == smile]
        test_df = pd.concat([test_df, test_row])
        df = df[df['amine'] != smile]

    return df, test_df
'''
Leave one out for the train / test splits.
    Args:
        df (dataframe): The pre-split dataframe.
        
        index (int): The index you want to leave out
                     for testing.
'''
def leaveoneout_splits(df, index):
    test = df[index:index+1]
    train = df.drop(index)

    return train, test

'''
Removing the acid fragment to reveal just the amine.
    Args:
        df (dataframe): The pre-split dataframe.
'''
def remove_acid(df):
    # The acid fragment smiles.
    acid = 'Clc1cc2c([C@H](C(Nc(cnc3)c4c3cccc4)=O)CN(CC=O)C2)cc1'
    acid = Chem.MolFromSmarts(acid)

    smiles = df['SMILES']
    peptides = [Chem.MolFromSmiles(s) for s in smiles]

    # Removing the acid fragment from the peptides.
    amines = []
    for mol in peptides:
        branch = AllChem.DeleteSubstructs(mol, acid)
        branch.UpdatePropertyCache()
        Chem.GetSymmSSSR(branch)
        amines.append(branch)

    df['Amine'] = amines
    return df

def main():
    # Importing the data.
    df = pd.read_csv('/Users/emmaking-smith/Moonshot/FINAL_CORRECTED_amide_coupling.csv', index_col=0)
    df = df.reset_index(drop=True)

    rbf_preds = []
    rbf_stds = []
    matern_preds = []
    matern_stds = []

    amine_rbf_preds = []
    amine_rbf_stds = []
    amine_matern_preds = []
    amine_matern_stds = []

    # Train / Test splits - Doing leave one out on entire dataframe.
    for i in range(len(df)):
        train_df, test_df = leaveoneout_splits(df, i)

        # Converting the amines to fingerprints.
        train_smiles = train_df['amine'].tolist()
        train_mols = []
        for s in train_smiles:
            # One SMILES string doesn't have the proper number of hydrogens on aromatic nitrogen.
            if s == 'Cc1cc(C)c(CN)c(=O)n1':
                s = 'Cc1cc(C)c(CN)c(=O)[nH]1'
            train_mols.append(Chem.MolFromSmiles(s))
        train_fingerprints = [Chem.RDKFingerprint(m) for m in train_mols]
        train_fingerprints = np.ravel(train_fingerprints).reshape(len(train_smiles), -1)

        test_smiles = test_df['amine'].to_list()
        # One SMILES string doesn't have the proper number of hydrogens on aromatic nitrogen.
        if test_smiles[0] == 'Cc1cc(C)c(CN)c(=O)n1':
            test_smiles[0] = 'Cc1cc(C)c(CN)c(=O)[nH]1'

        test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
        test_fingerprints = [Chem.RDKFingerprint(m) for m in test_mols]
        test_fingerprints = np.ravel(test_fingerprints).reshape(len(test_smiles), -1)

        # Retrieving the labels for the train / test sets.
        train_inhib = train_df['inhibition']

        # Gaussian Process training.
        rbf_gp.fit(train_fingerprints, train_inhib)
        matern_gp.fit(train_fingerprints, train_inhib)

        # Testing the gaussian processes on whole molecules.
        rbf_pred, rbf_std = rbf_gp.predict(test_fingerprints, return_std=True)
        print("rbf_pred", rbf_pred[0])
        print("rbf_std", rbf_std[0])
        matern_pred, matern_std = matern_gp.predict(test_fingerprints, return_std=True)
        print("matern_pred", matern_pred[0])
        print("matern_std", matern_std[0])
        # Appending Results to Lists.
        rbf_preds.append(rbf_pred[0])
        rbf_stds.append(rbf_std[0])
        matern_preds.append(matern_pred[0])
        matern_stds.append(matern_std[0])

    np.save('rbf_preds.npy', rbf_preds)
    np.save('rbf_stds.npy', rbf_stds)
    np.save('matern_preds.npy', matern_preds)
    np.save('matern_stds.npy', matern_stds)

    # RBF Plots.
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot([0, 1], [0, 1], '--', transform=ax.transAxes, color='gray')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('RBF Gaussian Process Regression (Leave One Out)')

    cmap = sns.cubehelix_palette(as_cmap=True)

    scatter_kwargs = {"zorder": 100}
    error_kwargs = {"lw": .5, "zorder": 0}

    points = plt.scatter(np.array(df['inhibition']), np.array(rbf_preds), s=20, c=np.array(df['yield']), cmap=cmap, **scatter_kwargs)
    plt.errorbar(np.array(df['inhibition']), np.array(rbf_preds), yerr=2*1.96*np.array(rbf_stds), fmt='None', ecolor='gray', **error_kwargs)

    # for i,xy in enumerate(zip(df['Inhibition'], rbf_preds)):
    #     ax.annotate(xy=xy, text=str(i))
    fig.colorbar(points)
    plt.savefig('correct_rbf.png')

    # Matern Plots (whole molecules).

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], '--', transform=ax.transAxes, color='gray')
    ax.set_xlabel('True Inhibition (%)')
    ax.set_ylabel('Pred Inhibition (%)')
    ax.set_title('Matern Gaussian Process Regression (Leave One Out)')

    cmap = sns.cubehelix_palette(as_cmap=True)

    scatter_kwargs = {"zorder": 100}
    error_kwargs = {"lw": .5, "zorder": 0}

    points = plt.scatter(np.array(df['inhibition']), np.array(matern_preds), s=20, c=np.array(df['yield']), cmap=cmap,
                         **scatter_kwargs)
    plt.errorbar(np.array(df['inhibition']), np.array(matern_preds), yerr=2 * 1.96 * np.array(matern_stds), fmt='None',
                 ecolor='gray', **error_kwargs)

    # for i,xy in enumerate(zip(df['Inhibition'], rbf_preds)):
    #     ax.annotate(xy=xy, text=str(i))
    fig.colorbar(points)
    plt.savefig('correct_matern.png')

if __name__ == '__main__':
    main()