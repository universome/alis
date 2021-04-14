"""
Runs a __reproducible__ experiment on the __current__ resources.
It copies the current codebase into a new project dir and runs an experiment from there.
"""
import sys; sys.path.extend(['.'])
import os
import argparse

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils import (
    create_project_dir,
    get_git_hash_suffix,
)


@hydra.main(config_name="../../configs/main.yml")
def main(cfg: DictConfig) -> str:
    for key in cfg:
        if isinstance(cfg[key], DictConfig) and '_target_' in cfg[key]:
            cfg[key] = instantiate(cfg[key])

    print("<=== CONFIG ===>")
    print(OmegaConf.to_yaml(cfg))

    before_train_cmd = '\n'.join(cfg.env.before_train_commands)
    torch_extensions_dir = os.environ.get('TORCH_EXTENSIONS_DIR', cfg.env.torch_extensions_dir)
    python_bin = cfg.env.python_bin
    training_cmd = f'{before_train_cmd}\n TORCH_EXTENSIONS_DIR={torch_extensions_dir} {python_bin} src/train.py {cfg.train_args_str}'

    print("<=== TRAINING COMMAND ===>")
    print(training_cmd)

    create_project_dir(cfg.project_release_dir, cfg.env.objects_to_copy, cfg.env.symlinks_to_create)
    os.chdir(cfg.project_release_dir)

    # Just in case, let's save the running command?
    with open(os.path.join(cfg.project_release_dir, 'training_cmd.sh'), 'w') as f:
        f.write(training_cmd + '\n')

    if cfg.print_only:
        print(training_cmd)
    else:
        os.system(training_cmd)



if __name__ == "__main__":
    main()