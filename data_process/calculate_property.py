import argparse
from data_process.utils import logP, QED, weight, SA, TPSA, get_mol, atom_valid_read,HBD, HBA, Rotatable
from tqdm import tqdm
import pandas as pd

def main(args):
    path_pool = [
        args.path_csv
    ]

    for path in path_pool:
        path_ori = path

        all = pd.read_csv(path_ori)
        target_path_csv = []
        fail = 0
        with tqdm(total=len(all)) as pbar:
            for index, row in all.iterrows():
                pbar.update(1)
                cur_mol = get_mol(row["SMILES"])

                if cur_mol == None or pd.isna(cur_mol) or not atom_valid_read(cur_mol):
                    target_path_csv.append([None] * 8)
                    fail += 1
                    continue

                cur_avilable_property_dict = {"TPSA": TPSA, "SA": SA,
                                              "Logp": logP, "QED": QED,
                                              "MolWt": weight,
                                              "HBD": HBD,
                                              "HBA": HBA,
                                              "rotatable": Rotatable
                                              }
                cur_mol_property_result = {key: cur_avilable_property_dict[key](cur_mol)
                                           for key in cur_avilable_property_dict.keys()}

                # if cur_mol_property_result["SA"] > 4.5:
                #     continue

                target_path_csv.append([
                                        cur_mol_property_result["Logp"],
                                        cur_mol_property_result["QED"],
                                        cur_mol_property_result["SA"],
                                        cur_mol_property_result["MolWt"],
                                        cur_mol_property_result["TPSA"],
                                        cur_mol_property_result["HBD"],
                                        cur_mol_property_result["HBA"],
                                        cur_mol_property_result["rotatable"]
                                        ])
        all["aLogP_label_continuous"] = [i[0] for i in target_path_csv]
        all["QED_label_continuous"] = [i[1] for i in target_path_csv]
        all["SAscore_label_continuous"] = [i[2] for i in target_path_csv]
        all["MolWt_label_continuous"] = [i[3] for i in target_path_csv]
        all["TPSA_label_continuous"] = [i[4] for i in target_path_csv]
        all["HBD"] = [i[5] for i in target_path_csv]
        all["HBA"] = [i[6] for i in target_path_csv]
        all["rotatable"] = [i[7] for i in target_path_csv]
        # save
        all.to_csv(path_ori, index=False)
        print("fail: {}".format(fail))
        print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_csv', type=str, default=None, required=True)
    args = parser.parse_args()

    main(args)
