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
def make_conditional_sample(sampler,
                            model,
                            tri_mode=True,
                            batch_size= 1,
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
                "various_conditions": torch.cat(batch_size * [torch.tensor(valid_list)[None, :]]).to(dtype=torch.float32,
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
def make_conditional_sample_mask_version(sampler, model, batch_size=1, custom_steps=None, eta=1.0,
                                         scale=1.0, property_set=None, uc_list=None, x0=None, mask=None,
                                         repaint_time=3, property_set_dict=None, uc_list_dict=None):
    log = dict()

    shape = [model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]
    t0 = time.time()

    with model.ema_scope():
        uc = None
        if scale != 1.0:
            uc_input = {
                "various_conditions": torch.cat(batch_size * [torch.tensor(uc_list)[None, :]]).to(dtype=torch.float32,
                                                                                                  device=model.device),
                "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(uc_list_dict)[None, :]]).to(
                    dtype=torch.bool, device=model.device),
            }
            uc = model.get_learned_conditioning(uc_input)
        # print(uc.shape)
        condition_input = {
            "various_conditions": torch.cat(batch_size * [torch.tensor(property_set)[None, :]]).to(dtype=torch.float32,
                                                                                                   device=model.device),
            "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(property_set_dict)[None, :]]).to(
                dtype=torch.bool, device=model.device),
        }
        c = model.get_learned_conditioning(condition_input)
        samples_ddim, _ = sampler.sample(S=custom_steps,
                                         conditioning=c,
                                         batch_size=batch_size,
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
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
    # print(f'Throughput for this batch: {log["throughput"]}')
    return log


def ori_scaffold_sidechain_exists(example):
    if not pd.isna(example["sidechain_number"]):
        if int(example["sidechain_number"]) > 0:
            return True
    return False


def valset_input_construct_helper_and_sample(input_df, cond_dict, sampler, model, property_constrained=2,
                                             batch_size=1,
                                             custom_steps=None,
                                             eta=1.0,
                                             scale=1.0, scale_pro=1.0, tri_mode=None
                                             ):
    # sample according to each line in input_df

    negvalue = [None, None,
                4.5, 0.3, None, 450., 100, 3, 7, 7
                ]

    input_property_set = []
    input_property_set_dict = []
    input_uc_list = []
    input_uc_list_dict = []
    input_valid_list = []
    input_valid_list_dict = []

    for row_index, row in input_df.iterrows():
        cur_example = [float(row["aLogP_label_continuous"]),
                       float(row["QED_label_continuous"]),
                       None,
                       float(row["MolWt_label_continuous"]),
                       float(row["TPSA_label_continuous"]),
                       float(row["HBD"]),
                       float(row["HBA"]),
                       float(row["rotatable"])
                       ]
        property_set, property_set_dict = property_masker(cur_example, cond_dict, property_constrained)
        if tri_mode:
            property_set = [cond_dict["None_valid_mol"], cond_dict["matched_property"]] + property_set
        else:
            property_set = [cond_dict["valid_mol"], cond_dict["matched_property"]] + property_set
        property_set_dict = [True, True] + property_set_dict

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
        valid_list = [cond_dict["valid_mol"]] + uc_list[1:]
        valid_list_dict = uc_list_dict

        # for property_set_dict_index in range(len(property_set_dict)):
        #     if property_set_dict[property_set_dict_index] == False:
        #         uc_list[property_set_dict_index] = negvalue[property_set_dict_index]
        #         uc_list_dict[property_set_dict_index] = False
        #         if tri_mode:
        #             valid_list[property_set_dict_index] = negvalue[property_set_dict_index]
        #             valid_list_dict[property_set_dict_index] = False

        input_property_set.append(property_set)
        input_property_set_dict.append(property_set_dict)
        input_uc_list.append(uc_list)
        input_uc_list_dict.append(uc_list_dict)
        input_valid_list.append(valid_list)
        input_valid_list_dict.append(valid_list_dict)

    uc_input = {
        "various_conditions": torch.tensor(input_uc_list).to(dtype=torch.float32,device=model.device),
        "various_conditions_discrete": torch.tensor(input_uc_list_dict).to(dtype=torch.bool, device=model.device)}
    valid_input = {
        "various_conditions": torch.tensor(input_valid_list).to(dtype=torch.float32, device=model.device),
        "various_conditions_discrete": torch.tensor(input_valid_list_dict).to(dtype=torch.bool, device=model.device)}
    condition_input = {
        "various_conditions": torch.tensor(input_property_set).to(dtype=torch.float32, device=model.device),
        "various_conditions_discrete": torch.tensor(input_property_set_dict).to(dtype=torch.bool, device=model.device)}

    log = dict()
    shape = [model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]
    t0 = time.time()
    with model.ema_scope():
        uc = model.get_learned_conditioning(uc_input)
        valid_c = model.get_learned_conditioning(valid_input)
        property_c = model.get_learned_conditioning(condition_input)

        if tri_mode:
            samples_ddim, _ = sampler.sample(S=custom_steps,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             eta=eta,
                                             triangle_sampling=tri_mode,
                                             unconditional_conditioning=uc,
                                             conditioning=valid_c,
                                             unconditional_guidance_scale=scale,
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


    return log, input_property_set, input_property_set_dict



def property_masker(cur_example, cond_dict, property_constrained):
    output_list_from_dict = [False] * 8
    property_mask = 7 - property_constrained
    property_pool = [0, 1, 3, 4, 5, 6, 7]
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
    none_index = random.sample(property_pool, property_mask)
    for tmp_index in none_index:
        cur_example[tmp_index] = None_value_list[tmp_index]
        output_list_from_dict[tmp_index] = True

    # del sa
    cur_example[2] = None_value_list[2]
    output_list_from_dict[2] = True

    return cur_example, output_list_from_dict


def run(model, imglogdir=None, logdir=None, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None,
        conditional_count=5, scale=1., condition_type=None, preset_str=None, scale_pro=1.,
        property_constrained=2, tri_mode=False):
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
            3.428, 0.6266, None, 300., 68., 0.0, 1.0, 1.0
        ]
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
                # uc_list[id + 2] = negvalue[id]
                # uc_list_dict[id + 2] = False
                # if tri_mode:
                #     # can change to input_value
                #     # depends on the task
                #     valid_list[id+2] = negvalue[id]
                #     valid_list_dict[id+2] = False
        # print("current negvalue is setting as {}".format(valid_list_dict))

        # you can give a neg direction like this:
        #   uc_list[3 + 2] = negvalue[3]
        #   uc_list[5 + 2] = negvalue[5]
        #   uc_list[6 + 2] = negvalue[6]
        #   uc_list[7 + 2] = negvalue[7]
        #   uc_list_dict[3 + 2] = False
        #   uc_list_dict[5 + 2] = False
        #   uc_list_dict[6 + 2] = False
        #   uc_list_dict[7 + 2] = False

        print("Your Sample condition is :")
        for i in range(len(extract_string)):
            if i == 2: continue
            print("{}:{}".format(keywords[i], extract_string[i]))

        print(f"Running conditional sampling for {n_samples} samples")
        print(f"each condition have {conditional_count} samples")
        print(f"so finally, output image is {conditional_count * n_samples}")
        for i in trange(n_samples, desc="Sampling Batches (conditional)"):
            logs = make_conditional_sample(sampler, model,
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
                                           scale_pro=scale_pro)
            cur_image_path = []
            for x_sample in logs["sample"]:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(imglogdir, f"{n_saved}.png"))
                cur_image_path.append(os.path.join(imglogdir, f"{n_saved}.png"))
                n_saved += 1
            for path in cur_image_path:
                final_image_results.append(property_set[2:] + property_set_dict[2:] + [path])

    elif condition_type == "mol_various_validation_from_dataset":
        evaluation_path = "select_csv_path.csv"
        evaluation_dataset = pd.read_csv(evaluation_path)
        # evaluation_dataset = evaluation_dataset[evaluation_dataset["MolWt_label_continuous"] > 350]
        # evaluation_dataset = evaluation_dataset[evaluation_dataset["MolWt_label_continuous"] > 300]
        # evaluation_dataset = evaluation_dataset[evaluation_dataset["MolWt_label_continuous"] > 250]
        # evaluation_dataset.reset_index(drop=True, inplace=True)
        evaluation_dataset = evaluation_dataset.sample(n=10000)
        evaluation_dataset.reset_index(drop=True, inplace=True)

        assert len(evaluation_dataset)%conditional_count == 0, "please make sure data can be divided by conditional_count"

        for i in trange(int(len(evaluation_dataset)/conditional_count), desc="Sampling Batches (conditional)"):
            cur_batch = evaluation_dataset[i*conditional_count:(i+1)*conditional_count]
            logs, property_set, property_set_dict = valset_input_construct_helper_and_sample(cur_batch, cond_dict, sampler, model, property_constrained=property_constrained,
                                             batch_size=conditional_count,
                                             custom_steps=custom_steps,
                                             eta=eta,
                                             scale=scale, scale_pro=scale_pro, tri_mode=tri_mode
                                             )

            cur_batch_index = 0
            for x_sample in logs["sample"]:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(imglogdir, f"{n_saved}.png"))
                os.path.join(imglogdir, f"{n_saved}.png")
                final_image_results.append(property_set[cur_batch_index][2:] + property_set_dict[cur_batch_index][2:] + [os.path.join(imglogdir, f"{n_saved}.png")])
                n_saved += 1
                cur_batch_index += 1


    target_image_path = pd.DataFrame(final_image_results, columns=["logp_setting", "QED_setting", "SA_setting",
                                                                   "MolWt_setting", "TPSA_setting", "HBD_setting",
                                                                   "HBA_setting", "rotatable_setting",
                                                                   "logp_None", "QED_None", "SA_None", "MolWt_None",
                                                                   "TPSA_None", "HBD_None", "HBA_None",
                                                                   "rotatable_None",
                                                                   "image_path"])
    target_image_path.to_csv(os.path.join(logdir, "image_path.csv"), index=False)
    print(f"path save to {logdir}/image_path.csv")
    print("done.")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load checkpoint in logdir",
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
        default=5
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
        help="valid gudience"
    )
    parser.add_argument(
        "--scale_pro",
        type=float,
        nargs="?",
        default=4,
        help="property gudience"
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
    )
    parser.add_argument(
        "-p",
        "--preset_str",
        type=str,
        default="",
    )
    parser.add_argument(
        "--property_num",
        type=int,
        default=2,
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

    run(model, imglogdir=imglogdir, eta=opt.eta, scale=opt.scale, scale_pro=opt.scale_pro, logdir=logdir,
        vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        conditional_count=opt.conditional_count,
        condition_type=opt.condition_type,
        preset_str=opt.preset_str,
        property_constrained=opt.property_num,
        tri_mode=opt.tri,)
