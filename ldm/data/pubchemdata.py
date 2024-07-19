import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import random
import torch


class pubchemBase(Dataset):
    def __init__(self,
                 csv_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 ):
        self.data_paths = csv_file
        data_csv = pd.read_csv(self.data_paths)  # !

        self.data_csv = data_csv
        self.labels = {
            "file_path_canimage": self.data_csv["Path"],
            "file_path_oriimage": self.data_csv["ori_path"],
            "file_path_scalimage": self.data_csv["scaffold_path"],
            "file_path_sidechainnumber": self.data_csv["sidechain_number"],
            "file_path_sidechainpath": self.data_csv["sidechain_path"]
        }

        self._length = len(self.data_csv)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        cur_example = dict()
        available_pool = [example["file_path_canimage"]]
        try:
            if not pd.isna(example["file_path_oriimage"]):
                available_pool.append(example["file_path_oriimage"])
                if not pd.isna(example["file_path_scalimage"]):
                    available_pool.append(example["file_path_scalimage"])
                    if int(example["file_path_sidechainnumber"]) > 0:
                        sidechain_path = example["file_path_sidechainpath"].split(",")
                        for sidechain_index in range(int(example["sidechain_number"])):
                            available_pool.append(sidechain_path[sidechain_index])
        except:
            pass
        cur_img_path = random.choice(available_pool)
        image = Image.open(cur_img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        cur_example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return cur_example


class pubchemBase_RL(Dataset):
    def __init__(self,
                 csv_file,
                 size=None,
                 interpolation="bicubic",
                 invalid_mode=False,
                 sampled_invalid_image_path=None,
                 ):
        self.data_paths = csv_file
        data_csv = pd.read_csv(self.data_paths)  # !
        self.invalid_mode = invalid_mode
        self.sampled_invalid_image_path = sampled_invalid_image_path
        if self.invalid_mode:
            self.data_csv = data_csv[['Path', 'Invalid_Image_pool']].astype(str)
            self.labels = {
                "file_path": self.data_csv["Path"],
                "Invalid_Image_pool": self.data_csv["Invalid_Image_pool"],
            }
            if self.sampled_invalid_image_path is not None:
                print("load sampled invalid image")
                self.sampled_invalid_image = pd.read_csv(self.sampled_invalid_image_path)["image_path"].tolist()
                print("it has :", len(self.sampled_invalid_image), "images")
        else:
            self.data_csv = data_csv[['Path']].astype(str)
            self.labels = {
                "file_path": self.data_csv["Path"],
            }

        del data_csv
        self._length = len(self.data_csv)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path"])

        output_example = dict()

        # invalid molecule
        # Can this kind of invalid data protect the integrity of the image?
        if self.invalid_mode and random.random() < 0.3:
            if self.sampled_invalid_image_path is not None and random.random() < 0.7:
                image = Image.open(random.choice(self.sampled_invalid_image))
            else:
                image = Image.open(random.choice(example["Invalid_Image_pool"].split(",")))
            output_example["mol_valid"] = np.array([0]).astype(np.float32)
        else:
            output_example["mol_valid"] = np.array([1]).astype(np.float32)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)

        output_example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        output_example["mol_valid_readable"] = "valid" if output_example["mol_valid"] == 1 else "invalid"

        return output_example

class pubchemBase_various_continuousV2(Dataset):
    def __init__(self,
                 csv_file=None,
                 size=None,
                 interpolation="bicubic",
                 invalid_mode=False,
                 sampled_invalid_image_path=None,
                 target_column=None,
                 rediscovery_mode=False
                 ):
        self.data_paths = csv_file
        self.sampled_invalid_image_path = sampled_invalid_image_path
        self.data_csv = pd.read_csv(self.data_paths)

        self.invalid_mode = invalid_mode
        if self.invalid_mode:
            self.labels = {
                "file_path_canimage": self.data_csv["Path"],
                "file_path_oriimage": self.data_csv["ori_path"],
                "file_path_scalimage": self.data_csv["scaffold_path"],
                "file_path_sidechainnumber": self.data_csv["sidechain_number"],
                "file_path_sidechainpath": self.data_csv["sidechain_path"],
                "Invalid_Image_pool": self.data_csv["Invalid_Image_pool"],
                "Logp": self.data_csv["aLogP_label_continuous"],
                "QED": self.data_csv["QED_label_continuous"],
                "SA": self.data_csv["SAscore_label_continuous"],
                "MolWt": self.data_csv["MolWt_label_continuous"],
                "TPSA": self.data_csv["TPSA_label_continuous"],
                "HBD": self.data_csv["HBD"],
                "HBA": self.data_csv["HBA"],
                "rotatable": self.data_csv["rotatable"]
            }
            if self.sampled_invalid_image_path is not None:
                print("load sampled invalid image from:", self.sampled_invalid_image_path)
                self.sampled_invalid_image = pd.read_csv(self.sampled_invalid_image_path)
                self.sampled_invalid_image_len = len(self.sampled_invalid_image)
                print("it has :", self.sampled_invalid_image_len)
        else:
            self.labels = {
                "file_path_canimage": self.data_csv["Path"],
                "Logp": self.data_csv["aLogP_label_continuous"],
                "QED": self.data_csv["QED_label_continuous"],
                "SA": self.data_csv["SAscore_label_continuous"],
                "MolWt": self.data_csv["MolWt_label_continuous"],
                "TPSA": self.data_csv["TPSA_label_continuous"],
                "HBD": self.data_csv["HBD"],
                "HBA": self.data_csv["HBA"],
                "rotatable": self.data_csv["rotatable"]
            }
        self._length = len(self.data_csv)

        self._cond_dict()
        self.property_interval_dict = self.property_interval_determine(self.labels)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def __len__(self):
        return self._length

    @staticmethod
    def build_dict():
        # various condition for molecular generation
        # for discrete property
        cond_dict = {}
        cond_dict["invalid_mol"] = 0
        cond_dict["valid_mol"] = 1
        cond_dict["None_valid_mol"] = 2
        cond_dict["unmatched_property"] = 3
        cond_dict["matched_property"] = 4
        cond_dict["None_property"] = 5
        cond_dict["None_logp"] = 6
        cond_dict["None_QED"] = 7
        cond_dict["None_SA"] = 8
        cond_dict["None_TPSA"] = 9
        cond_dict["None_MolWt"] = 10
        cond_dict["None_HBD"] = 11
        cond_dict["None_HBA"] = 12
        cond_dict["None_rotatable"] = 13

        cond_dict_valuetoname = {value: key for key, value in cond_dict.items()}
        return cond_dict, cond_dict_valuetoname

    @staticmethod
    def property_interval_determine(labels_dict):
        logp_interval = {"min": labels_dict["Logp"].min(), "max": labels_dict["Logp"].max()}
        qed_interval = {"min": labels_dict["QED"].min(), "max": labels_dict["QED"].max()}
        sa_interval = {"min": labels_dict["SA"].min(), "max": labels_dict["SA"].max()}
        molwt_interval = {"min": labels_dict["MolWt"].min(), "max": labels_dict["MolWt"].max()}
        tpsa_interval = {"min": labels_dict["TPSA"].min(), "max": labels_dict["TPSA"].max()}
        hbd_interval = {"min": labels_dict["HBD"].min(), "max": labels_dict["HBD"].max()}
        hba_interval = {"min": labels_dict["HBA"].min(), "max": labels_dict["HBA"].max()}
        rotatable_interval = {"min": labels_dict["rotatable"].min(), "max": labels_dict["rotatable"].max()}

        property_interval_dict = {"Logp": logp_interval,
                                  "QED": qed_interval,
                                  "SA": sa_interval,
                                  "MolWt": molwt_interval,
                                  "TPSA": tpsa_interval,
                                  "HBD": hbd_interval,
                                  "HBA": hba_interval,
                                  "rotatable": rotatable_interval
                                  }
        return property_interval_dict

    def _cond_dict(self):
        self.cond_dict, self.cond_dict_valuetoname = self.build_dict()

    def from_listvalue_to_string(self, cur_list, cur_list_from_dict):
        available_property_list = ["valid", "match", "Logp", "QED", "SA", "MolWt", "TPSA", "HBD", "HBA", "rotatable"]
        output_string = " "
        for i in range(len(available_property_list)):
            if cur_list_from_dict[i]:
                output_string += self.cond_dict_valuetoname[int(cur_list[i])] + " "
            else:
                output_string += available_property_list[i] + "_value:" + str(cur_list[i]) + " "
        return output_string

    @staticmethod
    def sampleproperty_to_list(cur_example, cond_dict, mask_mode=False, all_mask_mode=False):
        output_list = [cur_example["Logp"],
                       cur_example["QED"],
                       cond_dict["None_SA"],
                       cur_example["MolWt"],
                       cur_example["TPSA"],
                       cur_example["HBD"],
                       cur_example["HBA"],
                       cur_example["rotatable"],
                       ]
        output_list_from_dict = [False] * len(output_list)
        output_list_from_dict[2] = True

        if mask_mode:
            None_value_list = [
                cond_dict["None_logp"],
                cond_dict["None_QED"],
                cond_dict["None_SA"],
                cond_dict["None_MolWt"],
                cond_dict["None_TPSA"],
                cond_dict["None_HBD"],
                cond_dict["None_HBA"],
                cond_dict["None_rotatable"]
            ]
            none_times = random.randint(0, len(output_list) - 2)
            none_index = random.sample(range(len(output_list)), none_times)
            for tmp_index in none_index:
                output_list[tmp_index] = None_value_list[tmp_index]
                output_list_from_dict[tmp_index] = True

        if all_mask_mode:
            output_list = [
                cond_dict["None_logp"],
                cond_dict["None_QED"],
                cond_dict["None_SA"],
                cond_dict["None_MolWt"],
                cond_dict["None_TPSA"],
                cond_dict["None_HBD"],
                cond_dict["None_HBA"],
                cond_dict["None_rotatable"]
            ]
            output_list_from_dict = [True] * len(output_list)

        return output_list, output_list_from_dict

    @staticmethod
    def lets_mutate(property_range_start, property_range_end, cur_property, rejection_scale=0.2):
        property_range_len = abs(property_range_end - property_range_start)
        rejection_interval_start = cur_property - rejection_scale * property_range_len
        rejection_interval_end = cur_property + rejection_scale * property_range_len
        assert rejection_interval_start > property_range_start or rejection_interval_end < property_range_end, "rejection interval is out of range"
        mean_value = property_range_len / 2 + property_range_start
        while True:
            assume_property_mutate = np.random.normal(mean_value, property_range_len, 1)[0]
            if assume_property_mutate >= property_range_start and assume_property_mutate <= property_range_end:
                if assume_property_mutate > rejection_interval_end or assume_property_mutate < rejection_interval_start:
                    return assume_property_mutate

    @staticmethod
    def property_mutate(cur_list, index, property_interval_dict):
        available_property_list = ["Logp", "QED", "SA", "MolWt", "TPSA", "HBD", "HBA", "rotatable"]
        cur_property_value = cur_list[index]
        cur_property_name = available_property_list[index]
        cur_list[index] = pubchemBase_various_continuousV2.lets_mutate(property_interval_dict[cur_property_name]["min"],
                                                                       property_interval_dict[cur_property_name]["max"],
                                                                       cur_property_value)
        return cur_list

    @staticmethod
    def property_mutate_helper(cur_list, cur_property_list_dict, property_interval_dict):
        available_mutate_index = []
        for i in range(len(cur_property_list_dict)):
            if not cur_property_list_dict[i]:
                available_mutate_index.append(i)
        assert len(available_mutate_index) > 0, "No property can be mutated"

        mutate_times = random.randint(1, len(available_mutate_index))
        mutate_number = random.sample(available_mutate_index, mutate_times)
        for i in range(len(mutate_number)):
            cur_list = pubchemBase_various_continuousV2.property_mutate(cur_list, mutate_number[i],
                                                                        property_interval_dict)

        return cur_list

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        output_example = dict()

        list_for_property_condition = []
        list_for_whether_property_from_dict = []
        if self.invalid_mode:
            # three type of data construction
            # 30% random invalid images from Invalid_Image_pool
            # 70% random valid images from file_path_canimage/file_path_oriimage with its properties
            cur_dice = random.random()

            none_property_list = [self.cond_dict["None_logp"],
                                  self.cond_dict["None_QED"],
                                  self.cond_dict["None_SA"],
                                  self.cond_dict["None_MolWt"],
                                  self.cond_dict["None_TPSA"],
                                  self.cond_dict["None_HBD"],
                                  self.cond_dict["None_HBA"],
                                  self.cond_dict["None_rotatable"]
                                  ]
            none_property_from_dict = [True] * len(none_property_list)
            if cur_dice < 0.3:
                invalid_from_where_dice = random.random()
                # 20% random invalid images from Invalid_Image_pool   1:1:8 if resampled stage else 1:1:0
                if self.sampled_invalid_image_path is None or invalid_from_where_dice < 0.3:
                    if random.random() < 0.5:
                        # invalid mode1: random invalid mol images
                        cur_img_path = random.choice(example["Invalid_Image_pool"].split(","))
                        list_for_property_condition = [self.cond_dict["invalid_mol"],
                                                       self.cond_dict["None_property"]
                                                       ] + none_property_list
                        list_for_whether_property_from_dict = [True, True] + none_property_from_dict
                    else:
                        # invalid mode2: images with unmatched properties
                        available_pool = [example["file_path_canimage"]]
                        if not pd.isna(example["file_path_oriimage"]):
                            available_pool.append(example["file_path_oriimage"])
                        cur_img_path = random.choice(available_pool)
                        cur_property_list, cur_property_list_dict = self.sampleproperty_to_list(example,
                                                                                                self.cond_dict,
                                                                                                mask_mode=True)
                        # print("original_property_list: ", original_property_list)
                        mutate_property_list = self.property_mutate_helper(cur_property_list,
                                                                           cur_property_list_dict,
                                                                           self.property_interval_dict)
                        list_for_property_condition = [self.cond_dict["valid_mol"],
                                                       self.cond_dict["unmatched_property"]
                                                       ] + mutate_property_list
                        list_for_whether_property_from_dict = [True, True] + cur_property_list_dict
                else:
                    # invalid mode3: images generated from model itself
                    # the following setting depends on the invalid image type
                    # mainly two types 1. totally invalid molecular images
                    # 2. valid molecular images with invalid properties
                    # not used during the first training stage
                    # this label come from low_quality_image_various_conditions.py
                    temp_index = random.randint(0, self.sampled_invalid_image_len - 1)
                    temp_example = self.sampled_invalid_image.loc[temp_index, :]
                    cur_img_path = temp_example["image_path"]
                    available_property_label = ["aLogP_label", "QED_label", "SAscore_label", "MolWt_label",
                                                "TPSA_label", "HBD_label", "HBA_label", "rotatable_label"]
                    list_for_property_condition = []
                    list_for_whether_property_from_dict = []
                    for cur_label_index in range(len(available_property_label)):
                        if temp_example[available_property_label[cur_label_index] + "_None"]:
                            list_for_property_condition.append(none_property_list[cur_label_index])
                            list_for_whether_property_from_dict.append(True)
                        else:
                            list_for_property_condition.append(
                                float(temp_example[available_property_label[cur_label_index]]))
                            list_for_whether_property_from_dict.append(False)

                    if int(temp_example["property_match"]) == self.cond_dict["None_property"]:
                        prefix = [self.cond_dict["invalid_mol"], self.cond_dict["None_property"]]
                    elif int(temp_example["property_match"]) == self.cond_dict["unmatched_property"]:
                        prefix = [self.cond_dict["None_valid_mol"], self.cond_dict["unmatched_property"]]
                    else:
                        raise ValueError("Invalid property_match value: ", temp_example["property_match"])
                    list_for_property_condition = prefix + list_for_property_condition
                    list_for_whether_property_from_dict = [True, True] + list_for_whether_property_from_dict
            else:
                # 70% random valid images from file_path_canimage/file_path_oriimage with its properties
                available_pool = [example["file_path_canimage"]]
                if not pd.isna(example["file_path_oriimage"]):
                    available_pool.append(example["file_path_oriimage"])
                cur_img_path = random.choice(available_pool)
                cur_property_list, cur_property_list_dict = self.sampleproperty_to_list(example, self.cond_dict,
                                                                                        mask_mode=True)
                list_for_property_condition = [self.cond_dict["valid_mol"],
                                               self.cond_dict["matched_property"]
                                               ] + cur_property_list
                list_for_whether_property_from_dict = [True, True] + cur_property_list_dict
        else:
            cur_img_path = example["file_path_canimage"]
            cur_property_list, cur_property_list_dict = self.sampleproperty_to_list(example, self.cond_dict,
                                                                                    mask_mode=True)
            list_for_property_condition = [self.cond_dict["valid_mol"],
                                           self.cond_dict["matched_property"]
                                           ] + cur_property_list
            list_for_whether_property_from_dict = [True, True] + cur_property_list_dict

        # insert None value for whether valid and whether matched

        if list_for_property_condition[0] in [self.cond_dict["valid_mol"],
                                              self.cond_dict["invalid_mol"]] and random.random() < 0.2:
            list_for_property_condition[0] = self.cond_dict["None_valid_mol"]
        if list_for_property_condition[1] in [self.cond_dict["matched_property"],
                                              self.cond_dict["unmatched_property"]] and random.random() < 0.2:
            list_for_property_condition[1] = self.cond_dict["None_property"]

        output_example["various_conditions_continuous_readable"] = self.from_listvalue_to_string(
            list_for_property_condition, list_for_whether_property_from_dict)
        image = Image.open(cur_img_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)

        output_example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        output_example["various_conditions_continuous"] = {
            "various_conditions": np.array(list_for_property_condition).astype(np.float32),
            "various_conditions_discrete": np.array(list_for_whether_property_from_dict).astype(np.bool_)
        }
        return output_example


class pubchemBase_single_protein(pubchemBase_various_continuousV2):
    def __init__(self,
                 csv_file,
                 target_protein=None,
                 size=None,
                 interpolation="bicubic",
                 invalid_mode=False,
                 sampled_invalid_image_path=None,
                 sampled_expand_image_path=None,
                 rediscovery_mode=False,
                 ):
        protein_support_pool = ["CDK2", "AKT1", "EP2", "EP4", "ROCK1", "ROCK2", "HER2", "EGFR"]
        assert target_protein is not None, "target protein should be specified"
        assert target_protein in protein_support_pool, "target protein should be in the protein pool"

        self.target_protein = target_protein
        super(pubchemBase_single_protein, self).__init__(csv_file, size, interpolation, invalid_mode,
                                                  sampled_invalid_image_path, target_column=target_protein+"_activity",
                                                  rediscovery_mode=rediscovery_mode)

        self.labels[target_protein+"_activity"] = self.data_csv[target_protein+"_activity"] if (target_protein+"_activity") in self.data_csv.columns else [None] * self._length
        
        self.sampled_expand_image_path = sampled_expand_image_path
        if sampled_expand_image_path is not None:
            # data augmentation
            self.sampled_expand_image = pd.read_csv(sampled_expand_image_path)
            self.sampled_expand_image_len = len(self.sampled_expand_image)
            print("sampled_expand_image_len: ", self.sampled_expand_image_len)

    @staticmethod
    def build_dict():
        # various condition for molecular generation
        # for discrete property
        cond_dict = {}
        cond_dict["invalid_mol"] = 0
        cond_dict["valid_mol"] = 1
        cond_dict["None_valid_mol"] = 2
        cond_dict["unmatched_property"] = 3
        cond_dict["matched_property"] = 4
        cond_dict["None_property"] = 5
        cond_dict["None_logp"] = 6
        cond_dict["None_QED"] = 7
        cond_dict["None_SA"] = 8
        cond_dict["None_TPSA"] = 9
        cond_dict["None_MolWt"] = 10
        cond_dict["None_HBD"] = 11
        cond_dict["None_HBA"] = 12
        cond_dict["None_rotatable"] = 13

        cond_dict["unmatched_protein"] = 47
        cond_dict["matched_protein"] = 48
        cond_dict["None_protein"] = 49

        cond_dict["None_CDK2"] = 50
        cond_dict["Act_CDK2"] = 51
        cond_dict["None_CDK4"] = 53
        cond_dict["Act_CDK4"] = 54
        cond_dict["None_CDK6"] = 56
        cond_dict["Act_CDK6"] = 57
        cond_dict["None_AKT1"] = 59
        cond_dict["Act_AKT1"] = 60
        cond_dict["None_BTK"] = 62
        cond_dict["Act_BTK"] = 63
        cond_dict["None_EPHX2"] = 65
        cond_dict["Act_EPHX2"] = 66
        cond_dict["None_EP2"] = 68
        cond_dict["Act_EP2"] = 69
        cond_dict["None_EP4"] = 71
        cond_dict["Act_EP4"] = 72
        cond_dict["None_ROCK1"] = 74
        cond_dict["Act_ROCK1"] = 75
        cond_dict["None_ROCK2"] = 77
        cond_dict["Act_ROCK2"] = 78
        cond_dict["None_HER2"] = 80
        cond_dict["Act_HER2"] = 81
        cond_dict["None_EGFR"] = 83
        cond_dict["Act_EGFR"] = 84

        cond_dict_valuetoname = {value: key for key, value in cond_dict.items()}
        return cond_dict, cond_dict_valuetoname

    def from_listvalue_to_string(self, cur_list, cur_list_from_dict):
        available_property_list = ["valid", "match", "Logp", "QED", "SA", "MolWt", "TPSA", "HBD", "HBA", "rotatable"] + ["protein"] +  [self.target_protein]
        output_string = " "
        for i in range(len(available_property_list)):
            if cur_list_from_dict[i]:
                output_string += self.cond_dict_valuetoname[int(cur_list[i])] + " "
            else:
                output_string += available_property_list[i] + "_value:" + str(cur_list[i]) + " "
        return output_string

    @staticmethod
    def sample_protein_to_list(cur_example, protein_name, cond_dict, mask_mode=False):
        output_list = [cond_dict["matched_protein"]]
        if pd.isna(cur_example[protein_name + "_activity"]):
            output_list.append(cond_dict["None_" + protein_name])
        else:
            append_value = cond_dict["Act_" + protein_name]
            if mask_mode and random.random() < 0.2:
                append_value = cond_dict["None_" + protein_name]
                output_list[0] = cond_dict["None_protein"]
            output_list.append(append_value)
        output_list_from_dict = [True] * len(output_list)

        return output_list, output_list_from_dict

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        output_example = dict()

        if self.invalid_mode:
            cur_dice = random.random()
            none_property_list = [self.cond_dict["None_logp"],
                                  self.cond_dict["None_QED"],
                                  self.cond_dict["None_SA"],
                                  self.cond_dict["None_MolWt"],
                                  self.cond_dict["None_TPSA"],
                                  self.cond_dict["None_HBD"],
                                  self.cond_dict["None_HBA"],
                                  self.cond_dict["None_rotatable"]]

            none_protein_list  =  [self.cond_dict["None_{}".format(self.target_protein)]]

            none_property_from_dict = [True] * len(none_property_list)
            none_protein_from_dict = [True] * len(none_protein_list)
            if cur_dice < 0.3:
                invalid_from_where_dice = random.random()
                if self.sampled_invalid_image_path is None or invalid_from_where_dice < 0.1:
                    # invalid mode1: random invalid mol images
                    cur_img_path = random.choice(example["Invalid_Image_pool"].split(","))
                    list_for_property_condition = [self.cond_dict["invalid_mol"],
                                                   self.cond_dict["None_property"]
                                                   ] + none_property_list + \
                                                  [self.cond_dict["None_protein"]] + \
                                                  none_protein_list
                    list_for_whether_property_from_dict = [True, True] + none_property_from_dict + \
                                                          [True] + none_protein_from_dict
                else:
                    # invalid mode: images generated from model itself
                    # the following setting depends on the invalid image type
                    # mainly two types 1. totally invalid molecular images
                    # 2. valid molecular images with invalid properties/proteins

                    temp_index = random.randint(0, self.sampled_invalid_image_len - 1)
                    temp_example = self.sampled_invalid_image.loc[temp_index, :]
                    cur_img_path = temp_example["image_path"]
                    available_property_label = ["aLogP_label", "QED_label", "SAscore_label", "MolWt_label",
                                                "TPSA_label", "HBD_label", "HBA_label", "rotatable_label"]
                    list_for_property_condition = []
                    list_for_whether_property_from_dict = []
                    for cur_label_index in range(len(available_property_label)):
                        if temp_example[available_property_label[cur_label_index] + "_None"]:
                            list_for_property_condition.append(none_property_list[cur_label_index])
                            list_for_whether_property_from_dict.append(True)
                        else:
                            list_for_property_condition.append(
                                float(temp_example[available_property_label[cur_label_index]]))
                            list_for_whether_property_from_dict.append(False)

                    # for protein
                    list_for_protein_condition = [temp_example[self.target_protein+"_label"]]
                    list_for_whether_protein_from_dict = [True]

                    if int(temp_example["property_match"]) == self.cond_dict["None_property"]:
                        prefix = [self.cond_dict["invalid_mol"], self.cond_dict["None_property"]]
                        midfix = [self.cond_dict["None_protein"]]
                    elif int(temp_example["property_match"]) == self.cond_dict["unmatched_property"]:
                        prefix = [self.cond_dict["None_valid_mol"], self.cond_dict["unmatched_property"]]
                        midfix = [self.cond_dict["None_protein"]]
                    elif int(temp_example["protein_match"]) == self.cond_dict["unmatched_protein"]:
                        prefix = [self.cond_dict["None_valid_mol"], self.cond_dict["None_property"]]
                        midfix = [self.cond_dict["unmatched_protein"]]
                    else:
                        print("Invalid property_match value: ", temp_example["property_match"])
                        raise ValueError("Invalid property_match value: ", temp_example["property_match"])
                    list_for_property_condition = prefix + list_for_property_condition + \
                                                  midfix + list_for_protein_condition
                    list_for_whether_property_from_dict = [True, True] + list_for_whether_property_from_dict + \
                                                          [True] + list_for_whether_protein_from_dict
            else:
                # 60% random valid images from file_path_canimage/file_path_oriimage with its properties
                available_pool = [example["file_path_canimage"]]
                if not pd.isna(example["file_path_oriimage"]):
                    available_pool.append(example["file_path_oriimage"])
                cur_img_path = random.choice(available_pool)

                if self.sampled_expand_image_path is not None and random.random() < min(self.sampled_expand_image_len/(self._length + self.sampled_expand_image_len), 0.5):
                    # 50% images generated from model itself
                    temp_index = random.randint(0, self.sampled_expand_image_len - 1)
                    temp_example = self.sampled_expand_image.loc[temp_index, :]
                    cur_img_path = temp_example["image_path"]
                    available_property_label = ["aLogP_label", "QED_label", "SAscore_label", "MolWt_label",
                                                "TPSA_label", "HBD_label", "HBA_label", "rotatable_label"]
                    list_for_property_condition = []
                    list_for_whether_property_from_dict = []
                    for cur_label_index in range(len(available_property_label)):
                        if temp_example[available_property_label[cur_label_index] + "_None"]:
                            list_for_property_condition.append(none_property_list[cur_label_index])
                            list_for_whether_property_from_dict.append(True)
                        else:
                            list_for_property_condition.append(
                                float(temp_example[available_property_label[cur_label_index]]))
                            list_for_whether_property_from_dict.append(False)

                    # for protein
                    list_for_protein_condition = [temp_example[self.target_protein+"_label"] if random.random()>0.2 else self.cond_dict["None_{}".format(self.target_protein)] ]
                    list_for_whether_protein_from_dict = [True]

                    prefix = [self.cond_dict["None_valid_mol"], self.cond_dict["None_property"]]
                    midfix = [self.cond_dict["matched_protein"]]

                    list_for_property_condition = prefix + list_for_property_condition + \
                                                  midfix + list_for_protein_condition
                    list_for_whether_property_from_dict = [True, True] + list_for_whether_property_from_dict + \
                                                          [True] + list_for_whether_protein_from_dict
                else:
                    cur_property_list, cur_property_list_dict = self.sampleproperty_to_list(example, self.cond_dict,
                                                                                            mask_mode=True)

                    cur_protein_list, cur_protein_list_dict = self.sample_protein_to_list(example, self.target_protein,
                                                                                            self.cond_dict,
                                                                                            mask_mode=True)
                    list_for_property_condition = [self.cond_dict["valid_mol"],
                                                   self.cond_dict["matched_property"]
                                                   ] + cur_property_list + cur_protein_list
                    list_for_whether_property_from_dict = [True, True] + cur_property_list_dict + cur_protein_list_dict

        else:
            cur_img_path = example["file_path_canimage"]
            cur_property_list, cur_property_list_dict = self.sampleproperty_to_list(example, self.cond_dict,
                                                                                    all_mask_mode=True)
            cur_protein_list, cur_protein_list_dict = self.sample_protein_to_list(example, self.target_protein,
                                                                                    self.cond_dict,
                                                                                    mask_mode=True)
            list_for_property_condition = [self.cond_dict["valid_mol"],
                                           self.cond_dict["None_property"]
                                           ] + cur_property_list + cur_protein_list
            list_for_whether_property_from_dict = [True, True] + cur_property_list_dict + cur_protein_list_dict

        # insert None value for whether valid and whether matched

        if list_for_property_condition[0] in [self.cond_dict["valid_mol"],
                                              self.cond_dict["invalid_mol"]] and random.random() < 0.2:
            list_for_property_condition[0] = self.cond_dict["None_valid_mol"]
        if list_for_property_condition[1] in [self.cond_dict["matched_property"],
                                              self.cond_dict["unmatched_property"]] and random.random() < 0.2:
            list_for_property_condition[1] = self.cond_dict["None_property"]
        if list_for_property_condition[-2] in [self.cond_dict["matched_protein"],
                                              self.cond_dict["unmatched_protein"]] and random.random() < 0.2:
            list_for_property_condition[-2] = self.cond_dict["None_protein"]

        output_example["various_conditions_continuous_readable"] = self.from_listvalue_to_string(
            list_for_property_condition, list_for_whether_property_from_dict)
        image = Image.open(cur_img_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)

        output_example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        output_example["various_conditions_continuous"] = {
            "various_conditions": np.array(list_for_property_condition).astype(np.float32),
            "various_conditions_discrete": np.array(list_for_whether_property_from_dict).astype(np.bool_)
        }
        return output_example


class pubchem400wTrain(pubchemBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(csv_file="/your_path/your_csv.csv",
                         flip_p=flip_p,  **kwargs)


class pubchem400wValidation(pubchemBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(csv_file="/your_path/your_csv.csv",
                         flip_p=flip_p,  **kwargs)


class pubchem400wTrain_RL(pubchemBase_RL):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(csv_file="/your_path/mini_dataset_train.csv",
                         **kwargs)


class pubchem400wValidation_RL(pubchemBase_RL):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(csv_file="/your_path/mini_dataset_train.csv",
                          **kwargs)


class pubchem400wTrain_various_continuousV2(pubchemBase_various_continuousV2):
    def __init__(self, **kwargs):
        super().__init__(csv_file="/your_path/pubchem_400w_train.csv",
                         invalid_mode=True, **kwargs)


class pubchem400wValidation_various_continuousV2(pubchemBase_various_continuousV2):
    def __init__(self, **kwargs):
        super().__init__(csv_file="/your_path/pubchem_400w_val.csv",
                         **kwargs)

# protein
class pubchem400wTrain_single_protein(pubchemBase_single_protein):
    def __init__(self, **kwargs):
        super().__init__(csv_file="/your_path/actives.csv",
                        invalid_mode=True, **kwargs)

class pubchem400wValidation_single_protein(pubchemBase_single_protein):
    def __init__(self, **kwargs):
        super().__init__(csv_file="/your_path/actives.csv",
                         **kwargs)
