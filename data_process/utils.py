from rdkit import Chem
import pandas as pd
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors
from data_process.SA_Score import sascorer
from rdkit.Chem import rdMolDescriptors

def logP(mol):
    """
    Computes RDKit's logP
    """
    try:
        return Chem.Crippen.MolLogP(mol)
    except:
        return None

def QED(mol):
    """
    Computes RDKit's QED score
    """
    try:
        return qed(mol)
    except:
        return None

def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    try:
        return Descriptors.MolWt(mol)
    except:
        return None

def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    try:
        return sascorer.calculateScore(mol)
    except:
        return None

def TPSA(mol):
    try:
        return Chem.rdMolDescriptors.CalcTPSA(mol)
    except:
        return None

def HBD(mol):
    """
    """
    try:
        return rdMolDescriptors.CalcNumHBD(mol)
    except:
        return None


def HBA(mol):
    """
    """
    try:
        return rdMolDescriptors.CalcNumHBA(mol)
    except:
        return None

def Rotatable(mol):
    """
    """
    try:
        return rdMolDescriptors.CalcNumRotatableBonds(mol)
    except:
        return None

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)

        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        # !!!
        if not atom_valid_read(mol):
            return None
        return mol
    elif pd.isna(smiles_or_mol):
        return None
    return smiles_or_mol


def atom_valid_read(mol):
    allowed_atoms = {'C', 'N', 'O', 'Cl', 'H', 'S', 'F', 'Br', 'B', 'P', 'I'}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_atoms:
            return False
    return True