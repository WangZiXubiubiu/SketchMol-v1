import pandas as pd
import cairosvg
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem.Draw import DrawingOptions
DrawingOptions.atomLabelFontSize = 50
DrawingOptions.dotsPerAngstrom = 100
DrawingOptions.bondLineWidth = 5
from rdkit.Chem.rdmolops import GetDistanceMatrix
import argparse
import multiprocessing as mp
from functools import partial
import os
from tqdm import tqdm
from datetime import datetime
from rdkit.Chem.Draw import rdMolDraw2D


def find_middle_non_ring_bond(mol):
    rings = mol.GetRingInfo().BondRings()
    ring_bonds = {bond for ring in rings for bond in ring}
    best_bond = None
    best_diff = float('inf')
    for bond in mol.GetBonds():
        if bond.GetIdx() in ring_bonds:
            continue
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        dist_matrix = GetDistanceMatrix(mol)
        side1 = sum(dist_matrix[atom1] < dist_matrix[atom2])
        side2 = sum(dist_matrix[atom2] < dist_matrix[atom1])
        diff = abs(side1 - side2)
        if diff < best_diff:
            best_diff = diff
            best_bond = bond
    return best_bond.GetIdx(), best_bond

def get_smaller_substruct_atoms(mol, bond_idx):
    fragment = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
    frags = Chem.GetMolFrags(fragment, asMols=False)
    substruct_atoms = [set(frag) for frag in frags]
    smaller_substruct_atoms = min(substruct_atoms, key=len)
    return smaller_substruct_atoms

def split_molecule_at_bond(mol, bond_idx):
    fragment_smiles = Chem.FragmentOnBonds(mol, [bond_idx], dummyLabels=[(1, 1)])
    return Chem.MolToSmiles(fragment_smiles).split('.')

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
        return mol
    return smiles_or_mol

def image_save(ori_data, target_path):
    smiles_ori = ori_data["SMILES"]
    index = ori_data.index
    mol_path = []
    mol_path_split = []
    count = 0
    cur_mol = None
    pos = mp.current_process()._identity[0]
    with tqdm(total = len(smiles_ori), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for num in index:
            try:
                os.makedirs(target_path + "/{}".format(num), exist_ok=True)
                cur_target_path = target_path + "/{}".format(num)
                cur_mol = get_mol(smiles_ori[num])
                if cur_mol is None:
                    raise ValueError()
                Draw.MolToFile(cur_mol, target_path + "/{}.svg".format(num), wedgeBonds=False)
                cairosvg.svg2png(url=target_path + "/{}.svg".format(num),
                                 write_to=target_path + "/{}.png".format(num))
                os.remove(target_path + "images/" + "{}.svg".format(num))
                mol_path.append(target_path + "/{}.png".format(num))
                mol_path_split.append(None)

                # middle_bond_index, middle_bond = find_middle_non_ring_bond(cur_mol)
                # smaller_substruct_atoms = get_smaller_substruct_atoms(cur_mol, middle_bond_index)
                # bond_indices = []
                # startatom_middle_bond, endatom_middle_bond = middle_bond.GetBeginAtomIdx(), middle_bond.GetEndAtomIdx()
                # not_in_pool = [startatom_middle_bond, endatom_middle_bond]
                # smaller_substruct_atoms = smaller_substruct_atoms - set(not_in_pool)
                # for bond in cur_mol.GetBonds():
                #     if bond.GetBeginAtomIdx() in smaller_substruct_atoms and bond.GetEndAtomIdx() in smaller_substruct_atoms \
                #         and bond.GetBeginAtomIdx() not in not_in_pool and bond.GetEndAtomIdx() not in not_in_pool:
                #         bond_indices.append(bond.GetIdx())
                #
                # # {}_split.png images
                # drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
                # drawer.drawOptions().bondLineWidth = 3
                # drawer.drawOptions().atomLabelFontSize = 50
                # drawer.drawOptions().dotsPerAngstrom = 100
                #
                # highlight_color = (0, 0, 0)
                # drawer.DrawMolecule(
                #     cur_mol,
                #     highlightAtoms=smaller_substruct_atoms,
                #     highlightBonds=bond_indices,
                #     highlightAtomColors={atom: highlight_color for atom in smaller_substruct_atoms},
                #     highlightBondColors={bond: highlight_color for bond in bond_indices}
                # )
                # drawer.FinishDrawing()
                # output_file = cur_target_path + "/{}_split.svg".format(num)
                # with open(output_file, 'w') as f:
                #     f.write(drawer.GetDrawingText())
                # cairosvg.svg2png(url=cur_target_path + "/{}_split.svg".format(num),
                #                  write_to=cur_target_path + "/{}_split.png".format(num))
                # mol_path_split.append(cur_target_path + "/{}_split.png".format(num))
                #
                # # {}.png images
                # drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
                # drawer.drawOptions().bondLineWidth = 3
                # drawer.drawOptions().atomLabelFontSize = 50
                # drawer.drawOptions().dotsPerAngstrom = 100
                # highlight_color = (1, 1, 1)
                # drawer.DrawMolecule(
                #     cur_mol,
                #     highlightAtoms=smaller_substruct_atoms,
                #     highlightBonds=bond_indices,
                #     highlightAtomColors={atom: highlight_color for atom in smaller_substruct_atoms},
                #     highlightBondColors={bond: highlight_color for bond in bond_indices}
                # )
                # drawer.FinishDrawing()
                # output_file = cur_target_path + "/{}.svg".format(num)
                # with open(output_file, 'w') as f:
                #     f.write(drawer.GetDrawingText())
                # cairosvg.svg2png(url=cur_target_path + "/{}.svg".format(num),
                #                  write_to=cur_target_path + "/{}.png".format(num))
                # mol_path.append(cur_target_path + "/{}.png".format(num))

            # except ValueError:
            #     count = count + 1
            #     mol_path.append(cur_target_path + "/{}.png".format(num))
            #     mol_path_split.append(None)
            except:
                count = count + 1
                Draw.MolToFile(cur_mol, cur_target_path + "/{}.svg".format(num), wedgeBonds=False)
                cairosvg.svg2png(url=cur_target_path + "/{}.svg".format(num),
                                 write_to=cur_target_path + "/{}.png".format(num))
                # os.remove(cur_target_path + "images/" + "{}.svg".format(num))
                mol_path.append(cur_target_path + "/{}.png".format(num))
                mol_path_split.append(None)
                print(num)
            pbar.update()
    print("attention !!!!!! fail {} times".format(count))
    return {"all_path_images": mol_path, "all_path_images_split":mol_path_split}



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--path_ori', type=str, default="")

    return parser


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()

    path_ori = config.path_ori
    avil_train_dataset = "mol.csv"
    # avil_val_dataset = "validation.csv"
    target_image_dir = os.path.join(path_ori, "mol_image_dir")
    os.makedirs(target_image_dir, exist_ok=True)

    path_train_data = os.path.join(path_ori, avil_train_dataset)

    ori_train_data = pd.read_csv(path_train_data)
    len_ori_train_data = len(ori_train_data)
    # ori_val_data = pd.read_csv(path_val_data)
    # len_ori_val_data = len(ori_val_data)

    # merged_df = pd.concat([ori_train_data, ori_val_data], axis=0)
    # len_ori_all_data = len_ori_train_data + len_ori_val_data
    merged_df = ori_train_data
    len_ori_all_data = len_ori_train_data

    process_data = merged_df.loc[:, ["SMILES"]]
    process_index = [i for i in range(len_ori_all_data)]
    process_data.insert(process_data.shape[1], 'index', process_index)
    process_data.set_index('index', inplace=True)

    print(f"[{datetime.now()}] Construcing molecule images to {target_image_dir}.")
    print(f"Number of workers: {config.num_workers}. Total number of CPUs: {mp.cpu_count()}.")
    print(f"Total: {len_ori_all_data} molecules.\n")

    batch_size = (len_ori_all_data - 1) // config.num_workers + 1
    batches = [process_data[i:i + batch_size] for i in range(0, len_ori_all_data, batch_size)]

    func = partial(image_save, target_path=target_image_dir)

    all_path_images = []
    all_path_images_split= []
    with mp.Pool(config.num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
        for path_images in pool.imap(func, batches):
            all_path_images += path_images["all_path_images"]
            all_path_images_split += path_images["all_path_images_split"]

    train_path_images = all_path_images[:len_ori_train_data]
    train_path_images_split = all_path_images_split[:len_ori_train_data]
    # val_path_images = all_path_images[len_ori_train_data:]

    ori_train_data['Path'] = train_path_images
    ori_train_data["Path_split"] = train_path_images_split
    # ori_val_data.insert(ori_val_data.shape[1], 'Path', val_path_images)

    ori_train_data.to_csv(path_train_data, index=False)
    # ori_val_data.to_csv(path_val_data, index=False)

    print("{} is ok".format(path_train_data))
    # print("{} is ok".format(path_val_data))


