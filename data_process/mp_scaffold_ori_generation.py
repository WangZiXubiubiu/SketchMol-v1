from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import argparse
import numpy as np
from PIL import Image
from rdkit.Chem.Draw import DrawingOptions
import pandas as pd
from tqdm import tqdm
import cairosvg
from datetime import datetime
import multiprocessing as mp
from functools import partial



def static_char_num(Str):
    sub_structure_char_num_dict = {}
    for i in Str:
        sub_structure_char_num_dict[i] = Str.count(i)
    return sub_structure_char_num_dict

def find_sub_structure_index_list(sub_str, full_str):
    sub_structure_index_list = []
    b = full_str.count(sub_str)
    index = -1
    for i in range(b):
        index = full_str.find(sub_str, index + 1, len(full_str))
        sub_structure_index_list.append(index)
    return sub_structure_index_list


def image_save(ori_data, target_path):
    print("start process")
    scaffold_path = []
    sidechain_number = []
    sidechain_path = []
    ori_path = []

    ALL_SMILES = ori_data["SMILES"]
    index = ori_data.index

    fail_count = 0
    pos = mp.current_process()._identity[0]
    with tqdm(total = len(ALL_SMILES), desc=f"Processing {pos}", position=pos-1, ncols=80, leave=False) as pbar:
        for num in index:
            try:
                cur_scaffold_path, cur_sidechain_num, cur_sidechain_path, cur_ori_image_path = image_generation(
                    ALL_SMILES[num], target_path, num)
                cur_sidechain_path_str = ""
                for i in range(cur_sidechain_num):
                    cur_sidechain_path_str = cur_sidechain_path_str + cur_sidechain_path[i] + ","
                scaffold_path.append(cur_scaffold_path)
                sidechain_number.append(cur_sidechain_num)
                sidechain_path.append(cur_sidechain_path_str)
                ori_path.append(cur_ori_image_path)
            except:
                fail_count = fail_count + 1
                scaffold_path.append(pd.NA)
                sidechain_number.append(pd.NA)
                sidechain_path.append(pd.NA)
                ori_path.append(pd.NA)
            pbar.update(1)
    print("attention !!!!!! fail {} times".format(fail_count))
    return [scaffold_path, sidechain_number, sidechain_path, ori_path]


def image_generation(target_SMILES, target_path, cur_index):
    cur_path = os.path.join(target_path, "{}".format(cur_index))
    os.makedirs(cur_path, exist_ok=True)

    mol = Chem.MolFromSmiles(target_SMILES)
    Scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)

    SMILES_scaffold = Chem.MolToSmiles(Scaffold_mol)
    # print("Target SMILES scaffold is : ", SMILES_scaffold)

    core_smarts = SMILES_scaffold
    mol_target = Chem.MolFromSmiles(target_SMILES)
    mol_core = Chem.MolFromSmarts(core_smarts)
    assert mol_target.HasSubstructMatch(mol_core)

    sidechain_mol = Chem.ReplaceCore(mol_target, mol_core, labelByIndex=True)

    sidechain_frag_list = Chem.GetMolFrags(sidechain_mol, asMols=True)

    sidechain_smiles = [Chem.MolToSmiles(x) for x in sidechain_frag_list]
    sub_structure_all_smiles = []
    for i in range(len(sidechain_smiles)):
        try:
            sub_structure_current_smiles = sidechain_smiles[i].split("]")[1]
        except Exception as e:
            sub_structure_current_smiles = sidechain_smiles[i].split("*")[1]
        sub_structure_all_smiles.append(sub_structure_current_smiles)

    sub_structure_all_smiles.sort(key=len, reverse=True)
    count_in_smiles = [False for num in range(len(target_SMILES))]

    target_subtract_sub_structure_smiles_all = []
    head_tail_should_del_sub_structure_smiles_index = []
    for i in range(len(sub_structure_all_smiles)):
        sub_structure_char_num = static_char_num(sub_structure_all_smiles[i])
        head_N_char = target_SMILES[:len(sub_structure_all_smiles[i])]
        tail_N_char = target_SMILES[-len(sub_structure_all_smiles[i]):]
        head_N_char_num = static_char_num(head_N_char)
        tail_N_char_num = static_char_num(tail_N_char)

        if sub_structure_char_num == head_N_char_num:
            tmp_in_smiles = count_in_smiles[:len(sub_structure_all_smiles[i])]
            if True not in tmp_in_smiles:
                target_subtract_sub_structure_smiles = target_SMILES[len(sub_structure_all_smiles[i]):]
                target_subtract_sub_structure_smiles_all.append(target_subtract_sub_structure_smiles)
                head_tail_should_del_sub_structure_smiles_index.append(sub_structure_all_smiles[i])
                for num in range(len(sub_structure_all_smiles[i])):
                    count_in_smiles[num] = True
        if sub_structure_char_num == tail_N_char_num:
            tmp_in_smiles = count_in_smiles[-len(sub_structure_all_smiles[i]):]
            if True not in tmp_in_smiles:
                target_subtract_sub_structure_smiles = target_SMILES[:-len(sub_structure_all_smiles[i])]
                target_subtract_sub_structure_smiles_all.append(target_subtract_sub_structure_smiles)
                head_tail_should_del_sub_structure_smiles_index.append(sub_structure_all_smiles[i])
                for num in range(len(sub_structure_all_smiles[i])):
                    count_in_smiles[len(target_SMILES) - 1 - num] = True

    target_subtract_sub_structure_smiles_all = list(set(target_subtract_sub_structure_smiles_all))
    head_tail_should_del_sub_structure_smiles_index = list(set(head_tail_should_del_sub_structure_smiles_index))

    for i in range(len(head_tail_should_del_sub_structure_smiles_index)):
        sub_structure_all_smiles.remove(head_tail_should_del_sub_structure_smiles_index[i])

    full_str = target_SMILES

    all_sub_structure_index_list = []
    for i in range(len(sub_structure_all_smiles)):
        sub_str = sub_structure_all_smiles[i]
        sub_structure_index_list = find_sub_structure_index_list(sub_str, full_str)
        all_sub_structure_index_list.append(sub_structure_index_list)

    for i in range(len(all_sub_structure_index_list)):
        for j in range(len(all_sub_structure_index_list[i])):
            current_index = all_sub_structure_index_list[i][j]
            if (target_SMILES[current_index - 1] == '[' or
                target_SMILES[current_index - 1] == '(') and (
                target_SMILES[current_index + len(sub_structure_all_smiles[i])] == ']' or
                target_SMILES[current_index + len(sub_structure_all_smiles[i])] == ')'):
                tmp_in_smiles = count_in_smiles[current_index - 1: current_index + len(sub_structure_all_smiles[i])+1]
                if True not in tmp_in_smiles:
                    del_sub_structure_SMILES = target_SMILES[:current_index - 1] + target_SMILES[current_index + len(
                        sub_structure_all_smiles[i]) + 1:]
                    target_subtract_sub_structure_smiles_all.append(del_sub_structure_SMILES)
                    for num in range(len(sub_structure_all_smiles[i])+2):
                        count_in_smiles[current_index - 1 + num] = True
                    break           # one for one

    smis = target_subtract_sub_structure_smiles_all
    template_SMILES = SMILES_scaffold
    smis.append(target_SMILES)
    smis.append(template_SMILES)
    # smis.append(SMILES_scaffold)
    template = Chem.MolFromSmiles(template_SMILES)
    AllChem.Compute2DCoords(template)

    scaffold_path, sidechain_number, sidechain_path, ori_image_path = None, len(smis) - 2, [], None
    try:
        mols = []
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            AllChem.GenerateDepictionMatching2DStructure(mol, template)
            mols.append(mol)
        img_svg = Draw.MolsToGridImage(mols, molsPerRow=len(target_subtract_sub_structure_smiles_all),
                                   subImgSize=(256, 256), legends=['' for x in mols], useSVG=True)
        with open(os.path.join(cur_path, "all.csv"), 'w') as f:
            f.write(img_svg)
        cairosvg.svg2png(url=os.path.join(cur_path, "all.csv"), write_to=os.path.join(cur_path, "all.png"))

        img = Image.open(os.path.join(cur_path, "all.png"))
        row = 1
        column = len(smis)
        org_img = np.array(img)
        height, width = org_img.shape[:2]

        row_step = (int)(height / row)
        column_step = (int)(width / column)

        img = org_img[0:row_step * row, 0:column_step * column]

        for i in range(row):
            for j in range(column):
                tmp_img = img[(i * row_step):(i * row_step + row_step),
                          (j * column_step):(j * column_step) + column_step]
                tmp_img = Image.fromarray(tmp_img)
                if j == column - 1:
                    pic_name = "scaffold.png"
                    scaffold_path = os.path.join(cur_path, pic_name)
                elif j == column - 2:
                    pic_name = "ori.png"
                    ori_image_path = os.path.join(cur_path, pic_name)
                else:
                    pic_name = "side_chain_{}.png".format(j)
                    sidechain_path.append(os.path.join(cur_path, pic_name))
                # pic_name = './' + str(i) + "_" + str(j) + ".png"
                tmp_img.save(os.path.join(cur_path, pic_name))
    except Exception:
        # print(Exception)
        return scaffold_path, sidechain_number, sidechain_path, ori_image_path

    return scaffold_path, sidechain_number, sidechain_path, ori_image_path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--rerun', type=int, default=4)
    parser.add_argument('--path_ori', type=str, default="")

    return parser


def main():
    import warnings
    from rdkit import RDLogger
    warnings.filterwarnings("ignore")
    RDLogger.DisableLog('rdApp.error')

    parser = get_parser()
    config = parser.parse_args()

    path_ori = config.path_ori
    avil_dataset = "mol.csv"
    path_ori_data = os.path.join(path_ori, avil_dataset)
    ori_data = pd.read_csv(path_ori_data)

    process_data = ori_data.loc[:, ["SMILES"]]
    len_ori_all_data = len(process_data)
    process_index = [i for i in range(len_ori_all_data)]
    process_data.insert(process_data.shape[1], 'index', process_index)
    process_data.set_index('index', inplace=True)

    target_image_dir = "scaffold_sidechain_dir"
    target_image_dir = os.path.join(path_ori, target_image_dir)
    os.makedirs(target_image_dir, exist_ok=True)

    print(f"[{datetime.now()}] Construcing molecule images to {target_image_dir}.")
    print(f"Number of workers: {config.num_workers}. Total number of CPUs: {mp.cpu_count()}.")
    print(f"Total: {len_ori_all_data} molecules.\n")

    batch_size = (len_ori_all_data - 1) // (config.num_workers * config.rerun) + 1
    batches = [process_data[i:i + batch_size] for i in range(0, len_ori_all_data, batch_size)]
    func = partial(image_save, target_path=target_image_dir)

    # split the batches according to config.rerun
    rerun_size = (len(batches) - 1) // config.rerun + 1
    batches_split_rerun = [batches[i: i+rerun_size] for i in range(0, len(batches), rerun_size)]

    all_scaffold_path = []
    all_sidechain_number = []
    all_sidechain_path = []
    all_ori_path = []

    for cur_rerun in range(config.rerun):
        with mp.Pool(config.num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
            for path_images in pool.imap(func, batches_split_rerun[cur_rerun]):
                scaffold_path, sidechain_number, sidechain_path, ori_path = path_images
                all_scaffold_path = all_scaffold_path + scaffold_path
                all_sidechain_number = all_sidechain_number + sidechain_number
                all_sidechain_path = all_sidechain_path + sidechain_path
                all_ori_path = all_ori_path + ori_path

    ori_data['scaffold_path'] = all_scaffold_path
    ori_data['sidechain_number'] = all_sidechain_number
    ori_data['sidechain_path'] = all_sidechain_path
    ori_data['ori_path'] = all_ori_path
    ori_data.to_csv(path_ori_data, index=False)
    print("done!")
    print("save to", path_ori_data)

if __name__ == '__main__':
    main()
