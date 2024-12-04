"""
Common utility functions.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Utility functions
"""

import os
import sys

import torch
import yaml
import numpy as np
import random as python_random


def set_seed(options):
    np.random.seed(options["seed"])
    python_random.seed(options["seed"])
    torch.manual_seed(options["seed"])

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_next_model_folder(prefix, path = ''):
    model_folder = lambda prefix, run_idx: f"{prefix}_model_run_{run_idx}"

    # 在path下连续创建以prefix作为前缀的文件夹
    run_idx = 0
    while os.path.isdir(os.path.join(path, model_folder(prefix, run_idx))):
        run_idx += 1

    model_path = os.path.join(path, model_folder(prefix, run_idx))
    print(f"STARTING {prefix} RUN {run_idx}! Storing the models at {model_path}")

    return model_path


def set_dirs(config):
    """
    It sets up directory that will be used to save results.
    Directory structure:
          results -> model_mode_{} -> evaluation
                                   -> training -> loss
                                               -> model
                                               -> plots
    :return: None
    """
    # Update the config file with model config and flatten runtime config
    config = update_config_with_model(config)  # !!! 这一步其实已经可以关掉，因为在之前构建配置字典时已经运行过
    # Set main results directory using database name. Exp:  processed_data/dpp19
    paths = config["paths"]
    # data > processed_data
    processed_data_dir = os.path.join(paths["data"], "processed_data")
    # results > prefix_{}
    res_dir = get_next_model_folder(config["model_mode"], paths['results'])
    # results > prefix_{} > training
    training_dir = os.path.join(res_dir, "training")
    # results > prefix_{} > evaluation
    evaluation_dir = os.path.join(res_dir, "evaluation")
    # results > prefix_{} > training > model
    training_model_dir = os.path.join(training_dir, "model")
    # results > prefix_{} > training > plots
    training_plot_dir = os.path.join(training_dir, "plots")
    # results > prefix_{} > training > loss
    training_loss_dir = os.path.join(training_dir, "loss")
    # Create any missing directories
    create_dir(processed_data_dir)
    create_dir(evaluation_dir)
    create_dir(training_dir)
    create_dir(training_model_dir)
    create_dir(training_plot_dir)
    create_dir(training_loss_dir)
    # Print a message.
    config['paths'].update({'resDir': res_dir})
    print("Directories are set.")


def get_runtime_and_model_config():
    # 加载运行和模型配置文件，并以字典形式存储
    try:
        with open("./config/runtime.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading runtime config file")
    # Update the config by adding the model specific config to runtime config
    config = update_config_with_model(config)
    return config


def update_config_with_model(config):
    # 从运行配置中根据无监督学习方式加载模型配置，并将这些配置信息存入配置信息字典中
    model_config = config["unsupervised"]["model_mode"]
    try:
        with open("./config/"+model_config+".yaml", "r") as file:
            model_config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading model config file")
    config.update(model_config)
    # TODO: Clean up structure of configs
    # Add sub-category "unsupervised" as a flat hierarchy to the config:
    # 为了方便调用，将无监督学习的运行信息再次存放到配置字典中（其实就是简化检索过程）
    config.update(config["unsupervised"])
    return config


def update_config_with_model_dims(data_loader, config):
    ((xi, xj), _) = next(iter(data_loader))
    # Get the number of features
    dim = xi.shape[-1]
    # Update the dims of model architecture by adding the number of features as the first dimension
    config["dims"].insert(0, dim)
    return config

