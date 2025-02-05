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
from ldm.data.pubchemdata import pubchemBase_single_protein
import cv2
from scripts.inpaint_continuousV2 import make_batch


def extract_values(keywords, sequence):
    values = []

    for keyword in keywords:
        pattern = keyword + r":(-?\d*\.\d+|-?\d+)"
        match = re.search(pattern, sequence)

        if match:
            values.append(float(match.group(1)))
        else:
            values.append(None)

    return values


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
    log['mask'] = batch["mask"][0]
    log['ori_image'] = ori_image
    # log['ori_size_mask'] = batch['ori_size_mask']
    log['mask_for_visiual'] = batch["mask_for_visiual"]
    log['ori_image_decode'] = ori_image_decode

    log["sample"] = x_samples_ddim
    log["time"] = t1 - t0
    log['throughput'] = x_samples_ddim.shape[0] / (t1 - t0)
    return log


def run(model, imglogdir=None, logdir=None, vanilla=False, custom_steps=None, eta=None, n_samples=50000,
        conditional_count=5, scale=1., condition_type=None, preset_str=None, scale_pro=1., tri_mode=False,
        target_sample=0,  validation_dataset=None, mask_from_where=None, zoom_factor=None, repaint_time=3):

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(imglogdir, '*.png')))

    # dictionary build
    # from dataset method
    cond_dict, cond_dict_valuetoname = pubchemBase_single_protein.build_dict()
    sampler = DDIMSampler(model)
    final_image_results = []

    if condition_type == "mol_various_preset":
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
            cond_dict["None_rotatable"],

            cond_dict["None_protein"],
        ]
        uc_list_dict = [True] * len(uc_list)

        if tri_mode:
            property_set = [cond_dict["None_valid_mol"], cond_dict["matched_property"]]
            print("valid_scale:{}".format(float(scale)), "property_scale:{}".format(float(scale_pro)))
            valid_list = [cond_dict["valid_mol"]] + uc_list[1:]
            valid_list_dict = uc_list_dict
        else:
            property_set = [cond_dict["valid_mol"], cond_dict["matched_property"]]
            print("scale:{}".format(float(scale)))
            valid_list, valid_list_dict = None, None
        property_set_dict = [True, True]

        negvalue = [
            3.428, 0.6266, None, 366., 68., 1.0, 4.0, 5.0
        ]

        cur_string = preset_str
        assert cur_string != "", "please clarify your input"
        keywords = ["LogP", "QED", "sa", "MW", "TPSA", "HBD", "HBA", "RB",
                     "EP4", "AKT1", "ROCK1"]
        target_protein = None
        extract_string = extract_values(keywords, cur_string)
        for id, value in enumerate(extract_string):
            if id < 8:
                property_set.append(uc_list[id + 2])
                property_set_dict.append(True)
            else:
                if id == 8:
                    property_set.append(cond_dict["matched_protein"])
                    property_set_dict.append(True)
                    uc_list[id + 2] = cond_dict["unmatched_protein"]
                if value == None:
                    continue
                else:
                    target_protein = keywords[id]
                    property_set.append(cond_dict["Act_" + target_protein])
                    property_set_dict.append(True)
                    uc_list.append(cond_dict["Act_" + target_protein])
                    uc_list_dict.append(True)
                    valid_list.append(cond_dict["Act_" + target_protein])
                    valid_list_dict.append(True)
        print("Your Sample condition is :")
        for i in range(len(extract_string)):
            if i == 2: continue
            print("{}:{}".format(keywords[i], extract_string[i]))

        print(f"each condition have {conditional_count} samples")

        all_df_data = pd.read_csv(validation_dataset)
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
                final_image_results.append(property_set[2:10] + [property_set[11]] + property_set_dict[2:10] + [path])

            ori_image = 255. * rearrange(logs["ori_image"].cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(ori_image.astype(np.uint8)).save(os.path.join(cur_imglogdir, "ori.png"))
            cv2.imwrite(os.path.join(cur_imglogdir, "mask.png"), logs["mask_for_visiual"])
        target_image_path = pd.DataFrame(final_image_results, columns=["logp_setting", "QED_setting", "SA_setting",
                                                                   "MolWt_setting", "TPSA_setting", "HBD_setting",
                                                                   "HBA_setting", "rotatable_setting",
                                                                   "{}_setting".format(target_protein),
                                                                   "logp_None", "QED_None", "SA_None", "MolWt_None",
                                                                   "TPSA_None", "HBD_None", "HBA_None",
                                                                   "rotatable_None",
                                                                   "image_path"])
        target_image_path.to_csv(os.path.join(logdir, "image_path.csv"), index=False)
    elif condition_type == "mol_bio_change":
        cur_csv = pd.read_csv(validation_dataset)
        cur_csv = cur_csv[cur_csv["Path_split"].notna()]
        cur_csv = cur_csv.reset_index(drop=True)

        print("now we get {} samples".format(len(cur_csv)))

        target_task_pool = ["EP4", "ROCK1", "AKT1"]
        target_task = None
        for task in target_task_pool:
            if task.lower() in validation_dataset.lower():
                target_task = task
                break
        assert target_task is not None, "target task is not available"
        print("target task is {} optimization".format(target_task))

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
            cond_dict["None_rotatable"],
            cond_dict["unmatched_protein"],
            cond_dict["Act_" + target_task]
        ]
        uc_list_dict = [True] * len(uc_list)

        if tri_mode:
            property_set = [cond_dict["None_valid_mol"], cond_dict["None_property"]]
        else:
            property_set = [cond_dict["valid_mol"], cond_dict["None_property"]]
        property_set_dict = [True, True]

        property_post = [
            cond_dict["None_logp"],
            cond_dict["None_QED"],
            cond_dict["None_SA"],
            cond_dict["None_MolWt"],
            cond_dict["None_TPSA"],
            cond_dict["None_HBD"],
            cond_dict["None_HBA"],
            cond_dict["None_rotatable"],
            cond_dict["matched_protein"],
            cond_dict["Act_{}".format(target_task)]
        ]
        property_post_dict = [True] * len(property_post)

        property_set = property_set + property_post
        property_set_dict = property_set_dict + property_post_dict

        if tri_mode:
            print("valid_scale:{}".format(float(scale)), "property_scale:{}".format(float(scale_pro)))
            valid_list = [cond_dict["valid_mol"]] + uc_list[1:]
            valid_list[-2] = cond_dict["None_protein"]
            valid_list[-1] = cond_dict["Act_{}".format(target_task)]

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
            # cur_bio_active_rate = row["imagemol_predict_score"]

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
                    property_set[2:10] + [property_set[11]] + property_set_dict[2:10] + [cur_image_path, cur_smiles, target_task])
                # final_image_results.append(
                #     property_set[2:10] + [property_set[11]] + property_set_dict[2:10] + [cur_image_path, cur_smiles, cur_bio_active_rate, target_task])
            ori_image = 255. * rearrange(logs["ori_image"].cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(ori_image.astype(np.uint8)).save(os.path.join(cur_imglogdir, "ori.png"))

            cv2.imwrite(os.path.join(cur_imglogdir, "mask.png"), logs["mask_for_visiual"])

            # ori_image = 255. * rearrange(logs["ori_image_decode"].cpu().numpy(), 'c h w -> h w c')
            # Image.fromarray(ori_image.astype(np.uint8)).save(os.path.join(cur_imglogdir, "ori_image_decode.png"))

        target_image_path = pd.DataFrame(final_image_results, columns=["logp_setting", "QED_setting", "SA_setting",
                                                                   "MolWt_setting", "TPSA_setting", "HBD_setting",
                                                                   "HBA_setting", "rotatable_setting",
                                                                   "{}_setting".format(target_task),
                                                                   "logp_None", "QED_None", "SA_None", "MolWt_None",
                                                                   "TPSA_None", "HBD_None", "HBA_None",
                                                                   "rotatable_None",
                                                                   "image_path", "smiles", "target_protein"])
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
        default=10
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
        default=5
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
        default=2,
        help="unconditional gudience"
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
        # mol_various_preset
    )
    parser.add_argument(
        "-p",
        "--preset_str",
        type=str,
        default="",
    )
    parser.add_argument(
        "--mask_from_where",
        type=str,
        nargs="?",
        default="mol_various_preset"
    )
    parser.add_argument(
        "--zoom_factor",
        type=float,
        default=1,
        help="image zoom ratio 0.95-1.02 is great for most test cases, below is terrible"
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
        default="",
        nargs="?",
    )
    parser.add_argument(
        "--tri",
        type=bool,
        default=True,
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
        # global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        # global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

    model = load_model(config, ckpt, gpu, eval_mode)
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

    run(model, imglogdir=imglogdir, eta=opt.eta,
        scale=opt.scale, scale_pro=opt.scale_pro, logdir=logdir,
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
