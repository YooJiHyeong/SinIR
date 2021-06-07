import os
import sys
import argparse
import shutil
import logging
from glob import glob

from torchvision.utils import save_image

from loaders.loader import ImageHolder
from runners.runner import Runner
from utils.parser import Parser
from utils.runner_utils import denormalize, interp, interp_matlab


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('infer_dir_num', type=int)
    parser.add_argument('gpu', type=str)
    parser.add_argument('--rerun', '-rr', action='store_true')
    parser.add_argument('--yaml', '-y', type=str, required=True)
    p = parser.parse_args()

    return p.infer_dir_num, p.gpu, p.rerun, p.yaml


def set_logging(save_dir, gpu, rerun=False):
    os.makedirs(save_dir, exist_ok=rerun)

    log_format = f"%(asctime)s (GPU {gpu}: {save_dir}) %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="[%y/%m/%d %H:%M:%S]")

    fh = logging.FileHandler(f"{save_dir}/log.txt")
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def save_codes(save_dir):
    code_files = sorted(glob("./**/*.py", recursive=True))
    code_files = [path for path in code_files if '/outs/' not in path]
    for cf in code_files:
        dst = f"{save_dir}/dumped_codes/{os.path.dirname(cf)}"
        os.makedirs(dst, exist_ok=True)
        shutil.copy(cf, dst)


def make_save_dir(train_dir, yaml, img_shape):
    save_dir = []
    save_dir.append(f"[{os.path.basename(yaml.DATASET.img_path).split('.')[0]}]")
    save_dir.append("{}x{}".format(*img_shape))

    if yaml.USE_LAST_NET_ONLY:
        save_dir.append("LastNetOnly")

    for d in yaml.DESC:
        save_dir.append(d)

    return f"{train_dir}/{'_'.join(save_dir)}"


def infer_main(infer_dir_num=None, gpu=None, rerun=None, yaml=None):
    if all(v is None for v in [infer_dir_num, gpu, rerun, yaml]):
        infer_dir_num, gpu, rerun, yaml = arg_parse()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    train_dirs = []
    for f in glob(f"{os.getcwd()}/outs/*"):
        try:
            if int(os.path.basename(f)[:3]) == infer_dir_num:
                train_dirs.append(f)
        except Exception:
            continue
    assert len(train_dirs) == 1, "the folder does not exist or numbering is duplicated"
    train_dir = train_dirs[0]

    args_yaml_train = Parser(f"{train_dir}/args.yaml").C

    parser_infer = Parser(f"{os.getcwd()}/config_infer/{yaml}.yaml")
    args_yaml_infer = parser_infer.C

    if args_yaml_train.INTERP == 'pytorch':     # bicubic
        resampler = interp
    elif args_yaml_train.INTERP == 'matlab':    # bicubic + antialiasing
        resampler = interp_matlab

    img_holder = ImageHolder(**args_yaml_infer.DATASET,
                             resampler=resampler,
                             scale_num=args_yaml_train.DATASET.scale_num,
                             scale_factor=args_yaml_train.DATASET.scale_factor)
    infer_dir = make_save_dir(train_dir, args_yaml_infer, img_holder.img.shape[-2:])

    set_logging(infer_dir, gpu, rerun)
    shutil.copy(args_yaml_infer.DATASET.img_path, f"{infer_dir}/{os.path.basename(args_yaml_infer.DATASET.img_path)}")
    save_image(denormalize(img_holder.img), f"{infer_dir}/resized_{os.path.basename(args_yaml_infer.DATASET.img_path)}")
    shutil.copy(f"{os.getcwd()}/config_infer/{yaml}.yaml", f"{infer_dir}/{yaml}.yaml")

    logging.info(args_yaml_infer)
    logging.info(infer_dir)

    runner = Runner(img_holder,
                    train_dir,
                    **args_yaml_train.TRAIN,
                    **args_yaml_train.NET,
                    **args_yaml_train.OPT,
                    scale_num=img_holder.scale_num,
                    resampler=resampler)
    runner.load()
    runner.infer(infer_dir, img_holder.mask, args_yaml_infer.USE_LAST_NET_ONLY)


if __name__ == "__main__":
    infer_main()
