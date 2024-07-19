import argparse, os, sys, glob, datetime, yaml, random
import torch
import time
import numpy as np
from tqdm import trange
import pandas as pd
import re
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.data.pubchemdata import pubchemBase_various_continuousV2
import cv2
import PIL

def rescale_images_cv2(original_image, zoom_factor=0.8):
    if zoom_factor == 1:
        return original_image

    original_height, original_width = original_image.shape[:2]

    new_width = int(original_width * zoom_factor)
    new_height = int(original_height * zoom_factor)

    # Resize the image using cv2.resize
    resized_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC if zoom_factor > 1 else cv2.INTER_AREA)

    # # Ensure resized_image has 3 channels
    # if len(resized_image.shape) == 2 or resized_image.shape[2] == 1:
    #     resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
    if zoom_factor < 1:
        # Create a white background tensor if zooming out
        background_image = np.ones((original_height, original_width), dtype=np.uint8) * 255
        # Calculate the position to place the scaled image on the background
        x_offset = (original_width - new_width) // 2
        y_offset = (original_height - new_height) // 2
        # Place the resized image onto the background
        background_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        return background_image
    else:
        # Crop the central region of the image to the original size if zooming in
        x_offset = (new_width - original_width) // 2
        y_offset = (new_height - original_height) // 2
        cropped_image = resized_image[y_offset:y_offset + original_height, x_offset:x_offset + original_width]
        return cropped_image



def rescale_images_torch(original_image_tensor, zoom_factor=0.8):
    if zoom_factor == 1:
        return original_image_tensor

    # Assuming the original image tensor is in the format CxHxW
    channels, original_height, original_width = original_image_tensor.shape

    new_width = int(original_width * zoom_factor)
    new_height = int(original_height * zoom_factor)

    # Resize the image using torch.nn.functional.interpolate
    # We need to add two additional dimensions to the tensor: batch size and number of channels
    # to use interpolate, so we reshape the tensor to 1xCxHxW.
    resized_image_tensor = torch.nn.functional.interpolate(
        original_image_tensor.unsqueeze(0),  # Add batch dimension
        size=(new_height, new_width),  # New size
        mode='bicubic'  # Using area interpolation
    ).squeeze(0)  # Remove batch dimension after interpolation

    if zoom_factor < 1:
        # Create a white background tensor if zooming out
        background_tensor = torch.ones(channels, original_height, original_width, dtype=original_image_tensor.dtype) * 255
        # Calculate the position to place the scaled image on the background
        y_offset = (original_height - new_height) // 2
        x_offset = (original_width - new_width) // 2
        # Place the resized image onto the background
        background_tensor[:, y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image_tensor
        return background_tensor
    else:
        # Crop the central region of the image to the original size if zooming in
        y_offset = (new_height - original_height) // 2
        x_offset = (new_width - original_width) // 2
        cropped_image_tensor = resized_image_tensor[:, y_offset:y_offset + original_height, x_offset:x_offset + original_width]
        return cropped_image_tensor


def make_batch(input_image_path, batch_size, mask_shape, device="cuda", mask_from_where=None,
               df_data=None, zoom_factor = 1):
    input_shape_h, input_shape_w = 256, 256

    interpolation = PIL.Image.BICUBIC
    img = np.array(Image.open(input_image_path).resize((input_shape_h, input_shape_w), resample=interpolation))
    img = torch.tensor(img).to(torch.float32).transpose(0, 2).transpose(1, 2)
    img = rescale_images_torch(img, zoom_factor=zoom_factor) / 127.5 - 1.0
    image = img[None, ...].repeat(batch_size, 1, 1, 1)

    # if mask_from_where != "determined":
    #     # scaffold_image_path = df_data["scaffold_path"]
    #     sidechain_number = int(df_data["sidechain_number"])
    # else:
    #     # scaffold_image_path = None
    #     sidechain_number = 0

    if mask_from_where == "determined":
        b, h, w = batch_size, mask_shape[2], mask_shape[3]
        mask = torch.ones(b, h, w)

        dice = random.random()
        if dice < 0.5:
            dice2 = random.random()
            if dice2 < 1/4:
                mask[:, : , :w // 2] = 0.
            elif dice2 < 1/2:
                mask[:, : , w // 2:w] = 0.
            elif dice2 < 3/4:
                mask[:, :h//2 , :] = 0.
            else:
                mask[:, h // 2:h , :] = 0.
        else:
        #random mask a square area square is 1/4 of the whole image
            side_len = min(h, w) // 2
            # Randomly choose the top left corner of the square
            top = random.randint(0, h - side_len)
            left = random.randint(0, w - side_len)
            # Apply the mask
            mask[:, top:top + side_len, left:left + side_len] = 0.
            # zeros will be filled in
        mask = mask[:, None, ...]

    elif mask_from_where == "molecule_split":
        assert input_image_path is not None, "ori image path is None"

        ori_image = cv2.imread(input_image_path)
        ori_image = cv2.resize(ori_image, (input_shape_h, input_shape_w), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
        _, ori_binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        ori_binary = 255 - ori_binary
        kernel_size = 40
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        ori_dilated = cv2.dilate(ori_binary, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(ori_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if hierarchy[0][i][2] != -1:
                cv2.drawContours(ori_dilated, contours, i, (255, 255, 255), -1)

        highlight_image_path = df_data["Path_split"]
        highlight_img = cv2.imread(highlight_image_path)
        highlight_img = cv2.resize(highlight_img, (input_shape_h, input_shape_w), interpolation=cv2.INTER_AREA)
        highlight_img_gray = cv2.cvtColor(highlight_img, cv2.COLOR_BGR2GRAY)
        _, highlight_img_gray_binary = cv2.threshold(highlight_img_gray, 250, 255, cv2.THRESH_BINARY)
        highlight_img_gray_binary = 255 - highlight_img_gray_binary
        highlight_img_gray_binary = highlight_img_gray_binary - ori_binary

        kernel = np.ones((kernel_size+5, kernel_size+5), np.uint8)
        highlight_img_gray_binary = cv2.dilate(highlight_img_gray_binary, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(highlight_img_gray_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if hierarchy[0][i][2] != -1:
                cv2.drawContours(highlight_img_gray_binary, contours, i, (255, 255, 255), -1)

        before_connected_components = ori_dilated - highlight_img_gray_binary

        # restrain the num to 0/255
        _, before_connected_components = cv2.threshold(before_connected_components, 250, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(before_connected_components,
                                                                                4, cv2.CV_32S)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_mask = np.zeros_like(before_connected_components)
        new_mask[labels == largest_component] = 255

        kernel = np.ones((10, 10), np.uint8)

        new_mask = cv2.dilate(new_mask, kernel, iterations=1)

        new_mask = rescale_images_cv2(new_mask, zoom_factor=zoom_factor)
        ori_dilated = cv2.resize(new_mask, (mask_shape[2], mask_shape[3]), interpolation=cv2.INTER_AREA)
        dilated = torch.tensor(ori_dilated / 255).to(torch.float32)
        dilated = torch.where(dilated > 0, torch.tensor(1.0), dilated)

        mask = dilated[None, None, ...]

        batch = {"image": image.to(device), "mask": mask.to(device), "mask_for_visiual": new_mask}
    elif mask_from_where == "mol_various_preset":
        keep_region_image_path = df_data["Path_keep"]
        keep_region_image = cv2.imread(keep_region_image_path)
        keep_region_image = cv2.resize(keep_region_image, (input_shape_h, input_shape_w), interpolation=cv2.INTER_AREA)
        keep_region_image_gray = cv2.cvtColor(keep_region_image, cv2.COLOR_BGR2GRAY)
        _, keep_region_image_gray_binary = cv2.threshold(keep_region_image_gray, 250, 255, cv2.THRESH_BINARY)

        new_mask = rescale_images_cv2(keep_region_image_gray_binary, zoom_factor=zoom_factor)
        ori_dilated = cv2.resize(new_mask, (mask_shape[2], mask_shape[3]), interpolation=cv2.INTER_AREA)
        dilated = torch.tensor(ori_dilated / 255).to(torch.float32)
        # over 0 is 1
        dilated = torch.where(dilated > 0, torch.tensor(1.0), dilated)

        mask = dilated[None, None, ...]

        batch = {"image": image.to(device), "mask": mask.to(device), "mask_for_visiual": new_mask}

    return batch



def property_interval_determine():
    logp_interval = {"min": -2, "max": 7}
    qed_interval = {"min": 0, "max": 1}
    sa_interval = {"min": 1.5, "max": 4}
    molwt_interval = {"min": 50, "max": 600}
    tpsa_interval = {"min": 0, "max": 150}
    hbd_interval = {"min": 0, "max": 10}
    hba_interval = {"min": 0, "max":5}
    rotatable_interval = {"min": 0, "max": 10}

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


def extract_values(keywords, sequence):
    # 定义要搜索的关键词
    values = []

    for keyword in keywords:
        # 构建正则表达式模式
        pattern = keyword + r":(-?\d*\.\d+|-?\d+)"
        match = re.search(pattern, sequence)

        if match:
            # 将找到的值转换为浮点数
            values.append(float(match.group(1)))
        else:
            # 如果没有找到对应关键词，则返回 None
            values.append(None)

    return values


@torch.no_grad()
def make_conditional_sample(sampler,
                            model,
                            tri_mode=True,
                            batch_size=1,
                            custom_steps=None,
                            eta=1.0,
                            scale=1.0, scale_pro=1.0,
                            property_set=None,
                            property_set_dict=None,
                            uc_list=None,
                            uc_list_dict=None,
                            valid_list=None,
                            valid_list_dict=None,
                            ):
    log = dict()

    shape = [model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]
    t0 = time.time()

    with model.ema_scope():
        uc_input = {
            "various_conditions": torch.cat(batch_size * [torch.tensor(uc_list)[None, :]]).to(dtype=torch.float32,
                                                                                              device=model.device),
            "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(uc_list_dict)[None, :]]).to(
                dtype=torch.bool, device=model.device),
        }
        uc = model.get_learned_conditioning(uc_input)

        condition_input = {
            "various_conditions": torch.cat(batch_size * [torch.tensor(property_set)[None, :]]).to(dtype=torch.float32,
                                                                                                   device=model.device),
            "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(property_set_dict)[None, :]]).to(
                dtype=torch.bool, device=model.device),
        }
        property_c = model.get_learned_conditioning(condition_input)

        if tri_mode:
            valid_input = {
                "various_conditions": torch.cat(batch_size * [torch.tensor(valid_list)[None, :]]).to(
                    dtype=torch.float32,
                    device=model.device),
                "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(valid_list_dict)[None, :]]).to(
                    dtype=torch.bool, device=model.device),
            }
            valid_c = model.get_learned_conditioning(valid_input)
        else:
            valid_c = None

        if tri_mode:
            samples_ddim, _ = sampler.sample(S=custom_steps,
                                             conditioning=valid_c,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=eta,
                                             triangle_sampling=tri_mode,
                                             property_conditioning=property_c,
                                             property_condition_scale=scale_pro,
                                             )
        else:
            samples_ddim, _ = sampler.sample(S=custom_steps,
                                             conditioning=property_c,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             triangle_sampling=tri_mode,
                                             eta=eta)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    t1 = time.time()

    log["sample"] = x_samples_ddim
    log["time"] = t1 - t0
    log['throughput'] = x_samples_ddim.shape[0] / (t1 - t0)
    # print(f'Throughput for this batch: {log["throughput"]}')
    return log


@torch.no_grad()
def make_conditional_sample_mask_version(sampler,
                                         model,
                                         tri_mode=True,
                                         batch_size=1,
                                         custom_steps=None,
                                         eta=1.0,
                                         scale=1.0, scale_pro=1.0,
                                         property_set=None,
                                         property_set_dict=None,
                                         uc_list=None,
                                         uc_list_dict=None,
                                         valid_list=None,
                                         valid_list_dict=None,

                                         mask_from_where=None,
                                         df_data=None,
                                         repaint_time=3,
                                         zoom_factor=1,
                                         condition_type = None
                                         ):
    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]
    t0 = time.time()

    if condition_type in ["mol_property_change", "mol_various_preset"]:
        input_image = df_data["Path"]
    else:
        input_image = df_data["ori_path"]
    batch = make_batch(input_image, batch_size, mask_shape=shape, mask_from_where=mask_from_where,
                       df_data=df_data, zoom_factor=zoom_factor)

    with model.ema_scope():
        encoder_posterior = model.encode_first_stage(batch["image"])
        x0 = model.get_first_stage_encoding(encoder_posterior).detach()

        uc_input = {
            "various_conditions": torch.cat(batch_size * [torch.tensor(uc_list)[None, :]]).to(dtype=torch.float32,
                                                                                              device=model.device),
            "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(uc_list_dict)[None, :]]).to(
                dtype=torch.bool, device=model.device),
        }
        uc = model.get_learned_conditioning(uc_input)

        condition_input = {
            "various_conditions": torch.cat(batch_size * [torch.tensor(property_set)[None, :]]).to(dtype=torch.float32,
                                                                                                   device=model.device),
            "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(property_set_dict)[None, :]]).to(
                dtype=torch.bool, device=model.device),
        }
        property_c = model.get_learned_conditioning(condition_input)

        if tri_mode:
            valid_input = {
                "various_conditions": torch.cat(batch_size * [torch.tensor(valid_list)[None, :]]).to(
                    dtype=torch.float32,
                    device=model.device),
                "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(valid_list_dict)[None, :]]).to(
                    dtype=torch.bool, device=model.device),
            }
            valid_c = model.get_learned_conditioning(valid_input)
        else:
            valid_c = None

        if tri_mode:
            samples_ddim, _ = sampler.sample(S=custom_steps,
                                             conditioning=valid_c,
                                             batch_size=batch_size,
                                             shape=shape[1:],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=eta,
                                             triangle_sampling=tri_mode,
                                             property_conditioning=property_c,
                                             property_condition_scale=scale_pro,

                                             x0=x0,
                                             mask=batch["mask"],
                                             repaint=True,  # !!!!!!
                                             repaint_time=repaint_time
                                             )
        else:
            samples_ddim, _ = sampler.sample(S=custom_steps,
                                             conditioning=property_c,
                                             batch_size=batch_size,
                                             shape=shape[1:],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             triangle_sampling=tri_mode,
                                             eta=eta,

                                             x0=x0,
                                             mask=batch["mask"],
                                             repaint=True,  # !!!!!!
                                             repaint_time=repaint_time
                                             )

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        ori_image = torch.clamp((batch["image"][0] + 1.0) / 2.0, min=0.0, max=1.0)

        ori_image_decode = model.decode_first_stage(x0)
        ori_image_decode = torch.clamp((ori_image_decode[0] + 1.0) / 2.0, min=0.0, max=1.0)


    t1 = time.time()
    # log['mask'] = batch["mask"][0].to("cpu")
    log['ori_image'] = ori_image
    # log['ori_size_mask'] = batch['ori_size_mask']
    log['mask_for_visiual'] = batch["mask_for_visiual"]
    log['ori_image_decode'] = ori_image_decode

    log["sample"] = x_samples_ddim
    log["time"] = t1 - t0
    log['throughput'] = x_samples_ddim.shape[0] / (t1 - t0)
    return log


def ori_scaffold_sidechain_exists(example):
    if not pd.isna(example["sidechain_number"]):
        if int(example["sidechain_number"]) > 0:
            return True
    return False


def input_construct_helper_and_sample(input_df, cond_dict, sampler, model,
                                      batch_size=1,
                                      custom_steps=None,
                                      eta=1.0,
                                      scale=1.0, scale_pro=1.0,
                                      tri_mode = None,
                                      mask_from_where=None,
                                      zoom_factor=None,
                                      repaint_time=None
                                      ):
    # sample according to each line in input_df

    midvalue = [None, None,
                3.428, 0.6266, None, 366., 68., 1.0, 4.0, 5.0
                ]

    shape = [ batch_size,
            model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    input_property_set = []
    input_property_set_dict = []
    input_uc_list = []
    input_uc_list_dict = []
    input_valid_list = []
    input_valid_list_dict = []
    x0_list = []
    mask_list = []

    for row_index, row in input_df.iterrows():
        example_dict = {"Logp": float(row["aLogP_label_continuous"]),
                        "QED": float(row["QED_label_continuous"]),
                        "SA": float(row["SAscore_label_continuous"]),
                        "MolWt": float(row["MolWt_label_continuous"]),
                        "TPSA": float(row["TPSA_label_continuous"]),
                        "HBD": float(row["HBD"]),
                        "HBA": float(row["HBA"]),
                        "rotatable": float(row["rotatable"])
                        }
        property_extraction, property_extraction_from_dict = pubchemBase_various_continuousV2.sampleproperty_to_list(
            example_dict, cond_dict, mask_mode=True)

        if tri_mode:
            property_set = [cond_dict["None_valid_mol"], cond_dict["matched_property"]] + property_extraction
        else:
            property_set = [cond_dict["valid_mol"], cond_dict["matched_property"]] + property_extraction
        property_set_dict = [True, True] + property_extraction_from_dict

        uc_list = [
            cond_dict["None_valid_mol"],
            cond_dict["None_property"],
            cond_dict["None_logp"],
            cond_dict["None_QED"],
            cond_dict["None_SA"],
            cond_dict["None_MolWt"],
            cond_dict["None_TPSA"],
            cond_dict["None_HBD"],
            cond_dict["None_HBA"],
            cond_dict["None_rotatable"]
        ]
        uc_list_dict = [True] * len(uc_list)
        for property_set_dict_index in range(len(property_set_dict)):
            if property_set_dict[property_set_dict_index] == False:
                uc_list[property_set_dict_index] = midvalue[property_set_dict_index]
                uc_list_dict[property_set_dict_index] = False

        if tri_mode:
            valid_list = [cond_dict["valid_mol"]] + uc_list[1:]
            valid_list_dict = uc_list_dict
        else:
            valid_list, valid_list_dict = None, None

        input_property_set.append(property_set)
        input_property_set_dict.append(property_set_dict)
        input_uc_list.append(uc_list)
        input_uc_list_dict.append(uc_list_dict)
        input_valid_list.append(valid_list)
        input_valid_list_dict.append(valid_list_dict)

        if mask_from_where == "scaffold":
            input_image = row["ori_path"]
        else:
            # if row["ori_path"]非空 那就0.5的概率使用ori_path
            if not pd.isna(row["ori_path"]):
                input_image = row["ori_path"] if random.random() > 0.5 else row["Path"]
            else:
                input_image = row["Path"]
        x_0_mask = make_batch(input_image, batch_size=1,
                              mask_shape=shape, mask_from_where=mask_from_where,
                              df_data=row, zoom_factor=zoom_factor)
        x0_list.append(x_0_mask["image"])
        mask_list.append(x_0_mask["mask"])

    uc_input = {
        "various_conditions": torch.tensor(input_uc_list).to(dtype=torch.float32, device=model.device),
        "various_conditions_discrete": torch.tensor(input_uc_list_dict).to(dtype=torch.bool, device=model.device)}
    condition_input = {
        "various_conditions": torch.tensor(input_property_set).to(dtype=torch.float32, device=model.device),
        "various_conditions_discrete": torch.tensor(input_property_set_dict).to(dtype=torch.bool, device=model.device)}
    if tri_mode:
        valid_input = {
            "various_conditions": torch.tensor(input_valid_list).to(dtype=torch.float32, device=model.device),
            "various_conditions_discrete": torch.tensor(input_valid_list_dict).to(dtype=torch.bool, device=model.device)}

    # image combination
    x0_list = torch.cat(x0_list, dim=0)
    mask_list = torch.cat(mask_list, dim=0)

    log = dict()

    t0 = time.time()
    with model.ema_scope():
        encoder_posterior = model.encode_first_stage(x0_list)
        x0 = model.get_first_stage_encoding(encoder_posterior).detach()
        mask = mask_list

        uc = model.get_learned_conditioning(uc_input)
        property_c = model.get_learned_conditioning(condition_input)

        if tri_mode:
            valid_c = model.get_learned_conditioning(valid_input)

        if tri_mode:
            samples_ddim, _ = sampler.sample(S=custom_steps,
                                             conditioning=valid_c,
                                             batch_size=batch_size,
                                             shape=shape[1:],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=eta,
                                             triangle_sampling=tri_mode,
                                             property_conditioning=property_c,
                                             property_condition_scale=scale_pro,

                                             x0=x0,
                                             mask=mask,
                                             repaint=True,  # !!!!!!
                                             repaint_time=repaint_time
                                             )
        else:
            samples_ddim, _ = sampler.sample(S=custom_steps,
                                             conditioning=property_c,
                                             batch_size=batch_size,
                                             shape=shape[1:],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             triangle_sampling=tri_mode,
                                             eta=eta,

                                             x0=x0,
                                             mask=mask,
                                             repaint=True,  # !!!!!!
                                             repaint_time=repaint_time
                                             )

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    t1 = time.time()
    log["sample"] = x_samples_ddim
    log["time"] = t1 - t0
    log['throughput'] = x_samples_ddim.shape[0] / (t1 - t0)

    return log, input_property_set, input_property_set_dict


def run(model, imglogdir=None, logdir=None, vanilla=False, custom_steps=None, eta=None, n_samples=50000,
        conditional_count=5, scale=1., condition_type=None, preset_str=None, scale_pro=1., tri_mode=False,
        target_sample=0,
        validation_dataset=None, mask_from_where=None, zoom_factor=None, repaint_time=3):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(imglogdir, '*.png')))

    # dictionary build
    # from dataset method
    cond_dict, cond_dict_valuetoname = pubchemBase_various_continuousV2.build_dict()
    sampler = DDIMSampler(model)
    final_image_results = []

    midvalue = [None, None,
                3.428, 0.6266, None, 366., 68., 1.0, 4.0, 5.0
                ]
    if condition_type == "mol_various_preset":
        # rewrite uclisnt
        uc_list = [
            cond_dict["None_valid_mol"],
            cond_dict["None_property"],
            cond_dict["None_logp"],
            cond_dict["None_QED"],
            cond_dict["None_SA"],
            cond_dict["None_MolWt"],
            cond_dict["None_TPSA"],
            cond_dict["None_HBD"],
            cond_dict["None_HBA"],
            cond_dict["None_rotatable"]
        ]
        uc_list_dict = [True] * len(uc_list)

        if tri_mode:
            property_set = [cond_dict["valid_mol"], cond_dict["matched_property"]]
            print("valid_scale:{}".format(float(scale)), "property_scale:{}".format(float(scale_pro)))
            valid_list = [cond_dict["valid_mol"]] + uc_list[1:]
            valid_list_dict = uc_list_dict
        else:
            property_set = [cond_dict["valid_mol"], cond_dict["matched_property"]]
            print("scale:{}".format(float(scale)))
            valid_list, valid_list_dict = None, None
        property_set_dict = [True, True]

        midvalue = [
            3.428, 0.6266, None, 366., 68., 1.0, 4.0, 5.0
        ]

        # 输入序列 自动根据关键词读取LogP QED MW TPSA HBD HBA Rot
        cur_string = preset_str
        assert cur_string != "", "please clarify your input"
        keywords = ["LogP", "QED", "sa", "MW", "TPSA", "HBD", "HBA", "RB"]
        extract_string = extract_values(keywords, cur_string)

        for id, value in enumerate(extract_string):
            if value == None:
                property_set.append(uc_list[id + 2])
                property_set_dict.append(True)
            else:
                property_set.append(value)
                property_set_dict.append(False)
                uc_list[id + 2] = midvalue[id]
                uc_list_dict[id + 2] = False
                valid_list[id+2] = midvalue[id]
                valid_list_dict[id+2] = False

        print("Your Sample condition is :")
        for i in range(len(extract_string)):
            if i == 2: continue
            print("{}:{}".format(keywords[i], extract_string[i]))

        print(f"each condition have {conditional_count} samples")

        all_df_data = pd.read_csv(validation_dataset)
        smiles_data = None
        print("read from {}".format(validation_dataset))

        for i in trange(len(all_df_data), desc="Sampling Batches (conditional)"):
            df_data = all_df_data.iloc[i, :]
            cur_imglogdir = os.path.join(imglogdir, str(i))
            os.makedirs(cur_imglogdir, exist_ok=True)
            logs = make_conditional_sample_mask_version(sampler, model,
                                                        batch_size=conditional_count,
                                                        custom_steps=custom_steps,
                                                        eta=eta, scale=scale,
                                                        property_set=property_set,
                                                        property_set_dict=property_set_dict,
                                                        uc_list=uc_list,
                                                        uc_list_dict=uc_list_dict,
                                                        tri_mode=tri_mode,
                                                        valid_list=valid_list,
                                                        valid_list_dict=valid_list_dict,
                                                        scale_pro=scale_pro,
                                                        df_data=df_data,
                                                        mask_from_where=mask_from_where,
                                                        zoom_factor=zoom_factor,
                                                        repaint_time=repaint_time,
                                                        condition_type=condition_type)
            cur_image_path = []
            for x_sample in logs["sample"]:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(cur_imglogdir, f"{i}_{n_saved}.png"))
                cur_image_path.append(os.path.join(cur_imglogdir, f"{i}_{n_saved}.png"))
                n_saved += 1

            for path in cur_image_path:
                final_image_results.append(property_set[2:] + property_set_dict[2:] + [path, smiles_data])

            ori_image = 255. * rearrange(logs["ori_image"].cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(ori_image.astype(np.uint8)).save(os.path.join(cur_imglogdir, "ori.png"))
            cv2.imwrite(os.path.join(cur_imglogdir, "mask.png"), logs["mask_for_visiual"])

        target_image_path = pd.DataFrame(final_image_results, columns=["logp_setting", "QED_setting", "SA_setting",
                                                                       "MolWt_setting", "TPSA_setting", "HBD_setting",
                                                                       "HBA_setting", "rotatable_setting",
                                                                       "logp_None", "QED_None", "SA_None", "MolWt_None",
                                                                       "TPSA_None", "HBD_None", "HBA_None",
                                                                       "rotatable_None",
                                                                       "image_path", "ori_smiles"])
        target_image_path.to_csv(os.path.join(logdir, "image_path.csv"), index=False)
    # elif condition_type == "mol_various_random_from_dataset_mask_version":
        # print(f" output image is {conditional_count * n_samples}")
        #
        # training_path = "/work/data1/wangzixu/data/molecule_data/pubchem400w/final_400w_train.csv"
        # training_dataset = pd.read_csv(training_path)
        # if mask_from_where != "determined":
        #     # only keep  if column sidechain_path is not None
        #     training_dataset = training_dataset[training_dataset["sidechain_path"].notna()]
        #     # reset index
        #     training_dataset = training_dataset.reset_index(drop=True)
        #
        # if tri_mode:
        #     print("valid_scale:{}".format(float(scale)), "property_scale:{}".format(float(scale_pro)))
        # else:
        #     print("scale:{}".format(float(scale)))
        #
        # for i in trange(n_samples, desc="Sampling Batches (conditional)"):
        #     cur_imglogdir = os.path.join(imglogdir, str(i))
        #     os.makedirs(cur_imglogdir, exist_ok=True)
        #     # sample batch_size examples from training_dataset
        #     cur_batch = training_dataset.sample(n=conditional_count)
        #
        #     logs, input_property_set, input_property_set_dict = input_construct_helper_and_sample(cur_batch,
        #                                              cond_dict, sampler, model,
        #                                              batch_size=conditional_count,
        #                                              custom_steps=custom_steps,
        #                                              eta=eta, scale=scale,
        #                                              scale_pro=scale_pro,
        #                                              tri_mode=tri_mode,
        #                                              mask_from_where=mask_from_where,
        #                                              zoom_factor=zoom_factor,
        #                                              repaint_time=repaint_time)
        #
        #     for index, x_sample in enumerate(logs["sample"]):
        #         x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        #         Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(cur_imglogdir, f"{i}_{n_saved}.png"))
        #         cur_image_path = os.path.join(cur_imglogdir, f"{i}_{n_saved}.png")
        #         n_saved += 1
        #         final_image_results.append(input_property_set[index][2:] + input_property_set_dict[index][2:] + [cur_image_path, None])
        # target_image_path = pd.DataFrame(final_image_results, columns=["logp_setting", "QED_setting", "SA_setting",
        #                                                                "MolWt_setting", "TPSA_setting", "HBD_setting",
        #                                                                "HBA_setting", "rotatable_setting",
        #                                                                "logp_None", "QED_None", "SA_None", "MolWt_None",
        #                                                                "TPSA_None", "HBD_None", "HBA_None",
        #                                                                "rotatable_None",
        #                                                                "image_path", "ori_smiles"])
        # target_image_path.to_csv(os.path.join(logdir, "image_path.csv"), index=False)
    elif condition_type == "mol_property_change":
        cur_csv = pd.read_csv(validation_dataset)
        cur_csv = cur_csv[cur_csv["MolWt_label_continuous"] > 250]
        cur_csv = cur_csv[cur_csv["Path_split"].notna()]
        cur_csv = cur_csv.sample(n=100, random_state=42)
        cur_csv = cur_csv.reset_index(drop=True)

        print("now we get {} samples".format(len(cur_csv)))

        target_task_pool = ["Logp", "QED", "TPSA"]
        target_task = None
        for task in target_task_pool:
            if task.lower() in validation_dataset.lower():
                target_task = task
                break
        assert target_task is not None, "target task is not available"
        print("target task is {} optimization".format(target_task))

        # Logp QED TPSA
        # task_information = {"Logp": 4.34, "QED": 0.75, "TPSA": 50.0}
        task_pos = {"Logp": 0, "QED": 1, "TPSA": 4}
        task_column_name = {"Logp": "aLogP_label_continuous", "QED": "QED_label_continuous",
                            "TPSA": "TPSA_label_continuous"}
        property_interval_dict = property_interval_determine()
        property_change_value = abs(property_interval_dict[target_task]["max"] - property_interval_dict[target_task]["min"]) * 0.3

        uc_list = [
            cond_dict["None_valid_mol"],
            cond_dict["None_property"],
            cond_dict["None_logp"],
            cond_dict["None_QED"],
            cond_dict["None_SA"],
            cond_dict["None_MolWt"],
            cond_dict["None_TPSA"],
            cond_dict["None_HBD"],
            cond_dict["None_HBA"],
            cond_dict["None_rotatable"]
        ]
        uc_list_dict = [True] * len(uc_list)

        if tri_mode:
            property_set = [cond_dict["None_valid_mol"], cond_dict["matched_property"]]
        else:
            property_set = [cond_dict["valid_mol"], cond_dict["matched_property"]]
        property_set_dict = [True, True]

        property_post = [
            cond_dict["None_logp"],
            cond_dict["None_QED"],
            cond_dict["None_SA"],
            cond_dict["None_MolWt"],
            cond_dict["None_TPSA"],
            cond_dict["None_HBD"],
            cond_dict["None_HBA"],
            cond_dict["None_rotatable"]
        ]
        property_post_dict = [True] * len(property_post)

        property_post[task_pos[target_task]] = midvalue[task_pos[target_task] + 2]
        property_post_dict[task_pos[target_task]] = False
        property_set = property_set + property_post
        property_set_dict = property_set_dict + property_post_dict

        uc_list[task_pos[target_task] + 2] = midvalue[task_pos[target_task] + 2]
        uc_list_dict[task_pos[target_task] + 2] = False

        if tri_mode:
            print("valid_scale:{}".format(float(scale)), "property_scale:{}".format(float(scale_pro)))
            valid_list = [cond_dict["valid_mol"]] + uc_list[1:]
            valid_list_dict = property_set_dict
        else:
            print("scale:{}".format(float(scale)))
            valid_list, valid_list_dict = None, None


        print(f"Running conditional sampling for {n_samples} samples")
        print(f"each condition have {conditional_count} samples")
        print(f"so finally, output image is {conditional_count * n_samples}")

        final_image_results = []

        for i in trange(len(cur_csv), desc="Sampling Batches (conditional)"):
            cur_imglogdir = os.path.join(imglogdir, str(i))
            os.makedirs(cur_imglogdir, exist_ok=True)
            # get line
            row = cur_csv.iloc[i, :]
            cur_smiles = row["SMILES"]
            original_property = row[task_column_name[target_task]]

            if target_task in ["Logp"]:
                cur_target_change_value = original_property + property_change_value
            elif target_task in ["QED"]:
                cur_target_change_value = min(original_property + property_change_value, 0.95)
            elif target_task in ["TPSA"]:
                cur_target_change_value = max(original_property - property_change_value, 20)
            else:
                assert False, "target task is not available"

            property_set[task_pos[target_task] + 2] = cur_target_change_value

            logs = make_conditional_sample_mask_version(sampler, model,
                                                        batch_size=conditional_count,
                                                        custom_steps=custom_steps,
                                                        eta=eta, scale=scale,
                                                        property_set=property_set,
                                                        property_set_dict=property_set_dict,
                                                        uc_list=uc_list,
                                                        uc_list_dict=uc_list_dict,
                                                        tri_mode=tri_mode,
                                                        valid_list=valid_list,
                                                        valid_list_dict=valid_list_dict,
                                                        scale_pro=scale_pro,
                                                        df_data=row,
                                                        mask_from_where=mask_from_where,
                                                        zoom_factor=zoom_factor,
                                                        repaint_time=repaint_time,
                                                        condition_type="mol_property_change")
            for index, x_sample in enumerate(logs["sample"]):
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(cur_imglogdir, f"{i}_{n_saved}.png"))
                cur_image_path = os.path.join(cur_imglogdir, f"{i}_{n_saved}.png")
                n_saved += 1
                final_image_results.append(
                    property_set[2:] + property_set_dict[2:] + [cur_image_path, cur_smiles, original_property, task_column_name[target_task]])
            ori_image = 255. * rearrange(logs["ori_image"].cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(ori_image.astype(np.uint8)).save(os.path.join(cur_imglogdir, "ori.png"))

            # mask_sample = 255. * rearrange(logs["mask"].cpu().numpy(), 'c h w -> h w c')
            # mask_sample = np.repeat(mask_sample, 3, axis=2)
            # Image.fromarray(mask_sample.astype(np.uint8)).resize((256, 256), resample=PIL.Image.BICUBIC).save(os.path.join(cur_imglogdir, "mask.png"))
            cv2.imwrite(os.path.join(cur_imglogdir, "mask.png"), logs["mask_for_visiual"])

            ori_image = 255. * rearrange(logs["ori_image_decode"].cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(ori_image.astype(np.uint8)).save(os.path.join(cur_imglogdir, "ori_image_decode.png"))

            # cv2.imwrite(os.path.join(cur_imglogdir, "ori_size_mask.png"), logs["ori_size_mask"])

        target_image_path = pd.DataFrame(final_image_results, columns=["logp_setting", "QED_setting", "SA_setting",
                                                                       "MolWt_setting", "TPSA_setting", "HBD_setting",
                                                                       "HBA_setting", "rotatable_setting",
                                                                       "logp_None", "QED_None", "SA_None", "MolWt_None",
                                                                       "TPSA_None", "HBD_None", "HBA_None",
                                                                       "rotatable_None",
                                                                       "image_path", "ori_smiles", "original_property_value", "task_name"])
        target_image_path.to_csv(os.path.join(logdir, "image_path.csv"), index=False)





    print(f"path save to {logdir}/image_path.csv")
    print("done.")
    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )

    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=1
    )
    parser.add_argument(
        "--conditional_count",
        type=int,
        nargs="?",
        help="number of each smiles to draw",
        default=4
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "--target_sample",
        type=int,
        nargs="?",
        help="index of sample from validation dataset to draw",
        default=4
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        action='store_true',
        help="vanilla sampling: ddpm",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=250
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="?",
        default=2,
        help="unconditional gudience"
    )
    parser.add_argument(
        "--scale_pro",
        type=float,
        nargs="?",
        default=4,
        help=""
    )
    parser.add_argument(
        "--post",
        type=str,
        default="",
    )
    parser.add_argument(
        "--condition_type",
        type=str,
        default="mol_various_preset",
        # mol_various_preset, mol_various_random_from_dataset_mask_version
    )
    parser.add_argument(
        "-p",
        "--preset_str",
        type=str,
        default="",
    )
    parser.add_argument(
        "--proerty_num",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--tri",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--mask_from_where",
        type=str,
        nargs="?",
        default="molecule_split"
    )
    parser.add_argument(
        "--zoom_factor",
        type=float,
        default=1,
        help="image zoom ratio 0.9-1 is great for most test cases, below is terrible"
    )
    parser.add_argument(
        "--repaint_time",
        type=int,
        default=1,
        help="repaint time"
    )
    parser.add_argument(
        "--validation_dataset",
        type=str,
        nargs="?",
        default=""
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "last.ckpt")

    yaml_file = glob.glob(logdir + "/*.yaml")
    opt.base = yaml_file

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, now + opt.post, "samples")
    imglogdir = os.path.join(logdir, "img")
    # numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir, exist_ok=True)
    # os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    run(model, imglogdir=imglogdir, eta=opt.eta, scale=opt.scale, scale_pro=opt.scale_pro, logdir=logdir,
        vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        conditional_count=opt.conditional_count,
        condition_type=opt.condition_type,
        preset_str=opt.preset_str,
        tri_mode=opt.tri,
        target_sample=opt.target_sample,
        validation_dataset=opt.validation_dataset,
        mask_from_where=opt.mask_from_where,
        zoom_factor=opt.zoom_factor,
        repaint_time=opt.repaint_time,
        )
