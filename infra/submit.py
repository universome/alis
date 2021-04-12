#!/ibex/scratch/projects/c2112/envs/sgf/bin/python

"""
Run a __reproducible__ experiment on __allocated__ resources
It submits a slurm job(s) with the given hyperparams which will then execute `slurm_job.py`
This is the main entry-point
"""


import sys; sys.path.extend(['.', '..'])
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from utils import create_project_dir, BEFORE_TRAIN_CMD, PYTHON_BIN


@hydra.main(config_name="../configs/submit.yml")
def main(cfg: DictConfig):
    for key in cfg:
        if isinstance(cfg[key], DictConfig) and '_target_' in cfg[key]:
            cfg[key] = instantiate(cfg[key])

    assert Path(cfg.dataset.source_path).exists()

    # Setting up an experiment release directory
    outdir_arg = f'--outdir={cfg.project_release_dir}'
    training_cmd = f'{BEFORE_TRAIN_CMD}\n TORCH_EXTENSIONS_DIR={cfg.TORCH_EXTENSIONS_DIR} {PYTHON_BIN} scripts/train.py {cfg.training_cmd_cli_args_str}'

    print("<=== LAUNCH CONFIG ===>")
    print(OmegaConf.to_yaml(cfg))

    print('<=== TRAINING COMMAND ===>')
    print(training_cmd)

    create_project_dir(cfg.project_release_dir)

    os.chdir(cfg.project_release_dir)

    with open(os.path.join(cfg.project_release_dir, 'config_for_slurm_job.yml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    # Just in case, let's save the running command?
    with open(os.path.join(cfg.project_release_dir, 'training_cmd.sh'), 'w') as f:
        f.write(training_cmd + '\n')

    # Submitting the slurm job
    env_args_str = ','.join([f'{k}={v}' for k, v in cfg.env_args.items()])
    qos_arg_str = '--account conf-iccv-2021.03.25-elhosemh' if cfg.use_qos else ''
    submit_job_cmd = f'sbatch {cfg.sbatch_args_str} {qos_arg_str} --export=ALL,{env_args_str} infra/slurm_job.py'

    print('<=== SLURM COMMAND ===>')
    print(submit_job_cmd)

    if cfg.print_only:
        print(submit_job_cmd)
    else:
        os.system(submit_job_cmd)


if __name__ == "__main__":
    main()
