import argparse, os, sys, glob, datetime, yaml, random
import torch
import time
import numpy as np
from tqdm import trange
import pandas as pd

from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.data.pubchemdata import pubchemBase_single_protein

from sample_diffusion_condition_continuousV2 import extract_values

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True, steps=None,
               verbose=True,
               make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, )
    return samples, intermediates


@torch.no_grad()
def make_conditional_sample_tri_mode(sampler,
                                     model,
                                     batch_size=1,
                                     custom_steps=None,
                                     eta=1.0,
                                     scale=1.0,
                                     scale_pro=1.0,
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

        valid_input = {
            "various_conditions": torch.cat(batch_size * [torch.tensor(valid_list)[None, :]]).to(dtype=torch.float32,
                                                                                                 device=model.device),
            "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(valid_list_dict)[None, :]]).to(
                dtype=torch.bool, device=model.device),
        }
        valid_c = model.get_learned_conditioning(valid_input)
        # print(uc.shape)
        condition_input = {
            "various_conditions": torch.cat(batch_size * [torch.tensor(property_set)[None, :]]).to(dtype=torch.float32,
                                                                                                   device=model.device),
            "various_conditions_discrete": torch.cat(batch_size * [torch.tensor(property_set_dict)[None, :]]).to(
                dtype=torch.bool, device=model.device),
        }
        property_c = model.get_learned_conditioning(condition_input)
        samples_ddim, _ = sampler.sample(S=custom_steps,
                                         conditioning=valid_c,
                                         batch_size=batch_size,
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         eta=eta,
                                         triangle_sampling=True,
                                         property_conditioning=property_c,
                                         property_condition_scale=scale_pro,
                                         )

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    t1 = time.time()

    log["sample"] = x_samples_ddim
    log["time"] = t1 - t0
    log['throughput'] = x_samples_ddim.shape[0] / (t1 - t0)
    # print(f'Throughput for this batch: {log["throughput"]}')
    return log


@torch.no_grad()
def make_conditional_sample(sampler, model, batch_size=1, custom_steps=None, eta=1.0,
                            scale=1.0, property_set=None, property_set_dict=None, uc_list=None, uc_list_dict=None):
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


# run(model, imglogdir=imglogdir, eta=opt.eta, scale=opt.scale, logdir=logdir,
#         vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
#         conditional_count=opt.conditional_count,
#         condition_type=opt.condition_type, target_protein=opt.protein,
#         preset_str=opt.preset_str, tri_mode=opt.tri, scale_pro=opt.scale_pro)


def run(model, imglogdir=None, logdir=None, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None,
        conditional_count=5, scale=1, condition_type=None, target_protein=None, preset_str=None, scale_pro=1.,
        tri_mode=False):
    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(imglogdir, '*.png')))

    # dictionary build
    # from dataset method
    cond_dict, cond_dict_valuetoname = pubchemBase_single_protein.build_dict()
    sampler = DDIMSampler(model)
    final_image_results = []

    negvalue = [
        3.428, 0.6266, None, 366., 68., 1.0, 4.0, 5.0,
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

        cur_string = preset_str
        assert cur_string != "", "please clarify your input"
        keywords = ["LogP", "QED", "sa", "MW", "TPSA", "HBD", "HBA", "RB",
                     "EP4",  "AKT1", "ROCK1"]
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

        print(f"Running conditional sampling for {n_samples} samples")
        print(f"each condition have {conditional_count} samples")
        print(f"so finally, output image is {conditional_count * n_samples}")
        for i in trange(n_samples, desc="Sampling Batches (conditional)"):
            if tri_mode:
                logs = make_conditional_sample_tri_mode(sampler,
                                                        model,
                                                        batch_size=conditional_count,
                                                        custom_steps=custom_steps,
                                                        eta=eta,
                                                        scale=scale,
                                                        scale_pro=scale_pro,
                                                        property_set=property_set,
                                                        property_set_dict=property_set_dict,
                                                        uc_list=uc_list,
                                                        uc_list_dict=uc_list_dict,
                                                        valid_list=valid_list,
                                                        valid_list_dict=valid_list_dict,
                                                        )
            else:
                logs = make_conditional_sample(sampler,
                                               model,
                                               batch_size=conditional_count,
                                               custom_steps=custom_steps,
                                               eta=eta,
                                               scale=scale,
                                               property_set=property_set,
                                               property_set_dict=property_set_dict,
                                               uc_list=uc_list,
                                               uc_list_dict=uc_list_dict)

            cur_image_path = []
            for x_sample in logs["sample"]:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(imglogdir, f"{n_saved}.png"))
                cur_image_path.append(os.path.join(imglogdir, f"{n_saved}.png"))
                n_saved += 1
            for path in cur_image_path:
                final_image_results.append(property_set[2:10] + [property_set[11]] + property_set_dict[2:10] + [path])

    target_image_path = pd.DataFrame(final_image_results, columns=["logp_setting", "QED_setting", "SA_setting",
                                                                   "MolWt_setting", "TPSA_setting", "HBD_setting",
                                                                   "HBA_setting", "rotatable_setting",
                                                                   "{}_setting".format(target_protein),
                                                                   "logp_None", "QED_None", "SA_None", "MolWt_None",
                                                                   "TPSA_None", "HBD_None", "HBA_None",
                                                                   "rotatable_None",
                                                                   "image_path"])
    target_image_path.to_csv(os.path.join(logdir, "image_path.csv"), index=False)
    print(f"path save to {logdir}/image_path.csv")
    print("done.")
    # print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    cur_image_path = []
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    cur_image_path.append(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved, cur_image_path


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
        help="number of each sample to draw",
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
        help="unconditional gudience"
    )
    parser.add_argument(
        "--scale_pro",
        type=float,
        nargs="?",
        default=8,
        help="you may adjust this value for protein-constrained generation "
    )
    parser.add_argument(
        "--protein",
        type=str,
        help="which protein as a target to sample"
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
    # print(f"global step: {global_step}")
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

    run(model, imglogdir=imglogdir, eta=opt.eta, scale=opt.scale, logdir=logdir,
        vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        conditional_count=opt.conditional_count,
        condition_type=opt.condition_type, target_protein=opt.protein,
        preset_str=opt.preset_str, tri_mode=opt.tri, scale_pro=opt.scale_pro)
