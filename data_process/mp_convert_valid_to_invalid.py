import os as os
import pandas as pd
import argparse
from PIL import Image, ImageDraw
import numpy as np
from data_process.pylsd.pylsd.lsd import lsd
import random
from datetime import datetime
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import PIL

def image_save(ori_data, target_path, invalid_num):
    print("start process")
    index = ori_data.index
    mol_path = []

    fail_count = 0
    fail_read_ori_count = 0
    pos = mp.current_process()._identity[0]
    with tqdm(total=len(index), desc=f"Processing {pos}", position=pos - 1, ncols=80, leave=False) as pbar:
        for num in index:
            cur_path = os.path.join(target_path, "{}".format(num))
            os.makedirs(cur_path, exist_ok=True)
            cur_invalid_image_pool = []
            cur_valid_image_pool = []

            cur_index_data = ori_data.loc[num, :]

            cur_valid_image_pool.append(Image.open(cur_index_data["Path"]))
            try:
                # "scaffold_path", "sidechain_number", "sidechain_path", "ori_path"
                if not pd.isna(cur_index_data["ori_path"]):
                    cur_valid_image_pool.append(Image.open(cur_index_data["ori_path"]))
                    if not pd.isna(cur_index_data["scaffold_path"]):
                        cur_valid_image_pool.append(Image.open(cur_index_data["scaffold_path"]))
                        if int(cur_index_data["sidechain_number"]) > 0:
                            sidechain_path = cur_index_data["sidechain_path"].split(",")
                            for sidechain_index in range(int(cur_index_data["sidechain_number"])):
                                cur_valid_image_pool.append(Image.open(sidechain_path[sidechain_index]))
            except:
                fail_read_ori_count += 1
                print("error!!!!")

            for cur_invalid_num in range(invalid_num):
                try:
                    img = random.choice(cur_valid_image_pool)
                    img = img.resize((256, 256), resample=PIL.Image.BICUBIC)
                    gray = np.asarray(img.convert('L'))
                    lines = lsd(gray, num*invalid_num + cur_invalid_num)
                    draw = ImageDraw.Draw(img)
                    choice = random.uniform(0.01, 0.05)
                    for i in range(lines.shape[0]):
                        if random.random() < choice or i == lines.shape[0]//2:
                            pt1 = (int(lines[i, 0]), int(lines[i, 1]))
                            pt2 = (int(lines[i, 2]), int(lines[i, 3]))
                            width = lines[i, 4]
                            draw.line((pt1, pt2), fill=(255, 255, 255), width=int(np.ceil(2.25 * width)))
                    img.save(os.path.join(cur_path, '{}_{}.png'.format(num, cur_invalid_num)))
                    cur_invalid_image_pool.append(os.path.join(cur_path, '{}_{}.png'.format(num, cur_invalid_num)))
                except:
                    fail_count += 1

            cur_invalid_path_string = ",".join(cur_invalid_image_pool)
            mol_path.append(cur_invalid_path_string)
            pbar.update(1)

    print("attention !!!!!! fail ori {} times".format(fail_read_ori_count))

    print("attention !!!!!! fail {} times".format(fail_count))
    return mol_path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--invalid_count', type=int, default=5)
    parser.add_argument('--rerun', type=int, default=2)
    parser.add_argument('--path_ori', type=str, default="")

    return parser


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()

    path_ori = config.path_ori
    avil_dataset = "mol.csv"
    path_ori_data = os.path.join(path_ori, avil_dataset)
    ori_data = pd.read_csv(path_ori_data)
    target_image_dir = "invalid_images_dirv"
    target_image_dir = os.path.join(path_ori, target_image_dir)
    os.makedirs(target_image_dir, exist_ok=True)

    process_data = ori_data.loc[:, ["Path", "scaffold_path", "sidechain_number", "sidechain_path", "ori_path"]]
    len_ori_all_data = len(process_data)
    process_index = [i for i in range(len_ori_all_data)]
    process_data.insert(process_data.shape[1], 'index', process_index)
    process_data.set_index('index', inplace=True)

    print(f"[{datetime.now()}] Construcing molecule images to {target_image_dir}.")
    print(f"Number of workers: {config.num_workers}. Total number of CPUs: {mp.cpu_count()}.")
    print(f"Total: {len_ori_all_data} molecules.\n")

    batch_size = (len_ori_all_data - 1) // (config.num_workers * config.rerun) + 1
    batches = [process_data[i:i + batch_size] for i in range(0, len_ori_all_data, batch_size)]
    func = partial(image_save, target_path=target_image_dir, invalid_num=config.invalid_count)

    # split the batches according to config.rerun
    rerun_size = (len(batches) - 1) // config.rerun + 1
    batches_split_rerun = [batches[i: i+rerun_size] for i in range(0, len(batches), rerun_size)]

    # every ori_image got a pool
    all_path_images = []
    for cur_rerun in range(config.rerun):
        print("rerun {} start".format(cur_rerun))
        with mp.Pool(config.num_workers, initializer=tqdm.set_lock, initargs=(mp.RLock(),)) as pool:
            for path_images in pool.imap(func, batches_split_rerun[cur_rerun]):
                all_path_images = all_path_images + path_images

    ori_data['Invalid_Image_pool'] = all_path_images
    ori_data.to_csv(path_ori_data, index=False)
    print("{} is ok".format(path_ori_data))