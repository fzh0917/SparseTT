# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip
import argparse
import os.path as osp
from copy import deepcopy

import yaml
from loguru import logger

import torch

from main.test import build_sat_tester, build_sparsett_tester
from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.utils import complete_path_wt_root_in_cfg, hpo


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='experiments/sparsett/test/otb2015/sparsett_swin_otb2015.yaml',
                        type=str,
                        help='experiment configuration')
    parser.add_argument(
        '-hpocfg',
        '--hpo-config',
        default='experiments/sparsett/hpo/sparsett_swin_hpo_otb2015.yaml',
        type=str,
        help='experiment configuration')

    return parser


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg_origin = specify_task(root_cfg)

    # hpo config
    with open(parsed_args.hpo_config, "r") as f:
        hpo_cfg = yaml.safe_load(f)
    hpo_cfg = hpo_cfg["test"][task]
    hpo_schedules = hpo.parse_hp_path_and_range(hpo_cfg)

    csv_file = osp.join(hpo_cfg["exp_save"],
                        "hpo_{}.csv".format(task_cfg_origin["exp_name"]))

    torch.multiprocessing.set_start_method('spawn', force=True)

    while True:
        task_cfg = deepcopy(task_cfg_origin)
        hpo_exp_dict = hpo.sample_and_update_hps(task_cfg, hpo_schedules)
        if task == "track":
            testers = build_sparsett_tester(task_cfg)
        elif task == "vos":
            testers = build_sat_tester(task_cfg)
        else:
            logger.error("task {} is not supported".format(task_cfg))
            exit()
        task_cfg.freeze()
        tester = testers[0]
        test_result_dict = tester.test()
        hpo_exp_dict["main_performance"] = test_result_dict["main_performance"]
        df = hpo.dump_result_dict(csv_file, hpo_exp_dict)
        df.sort_values(by='main_performance', inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(df)
        del testers
