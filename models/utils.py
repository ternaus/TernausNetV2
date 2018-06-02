"""
Adaptation from https://github.com/mapillary/inplace_abn
"""

from functools import partial

import torch.nn as nn
from modules import ABN, InPlaceABNWrapper


def _get_norm_act(network_config):
    if network_config["bn_mode"] == "standard":
        if network_config["activation"] == "relu":
            return partial(ABN, activation=nn.ReLU(inplace=True))
        elif network_config["activation"] == "leaky_relu":
            return partial(ABN, activation=nn.LeakyReLU(network_config["leaky_relu_slope"], inplace=True))
        elif network_config["activation"] == "elu":
            return partial(ABN, activation=nn.ELU(inplace=True))
        else:
            print("Standard batch normalization is only compatible with relu, leaky_relu and elu")
            exit(1)
    elif network_config["bn_mode"] == "inplace":
        if network_config["activation"] == "leaky_relu":
            return partial(InPlaceABNWrapper, activation="leaky_relu", slope=network_config["leaky_relu_slope"])
        elif network_config["activation"] in ["elu", "none"]:
            return partial(InPlaceABNWrapper, activation=network_config["activation"])
        else:
            print("Inplace batch normalization is only compatible with leaky_relu, elu and none")
            exit(1)
    else:
        print("Unrecognized batch normalization mode", network_config["bn_mode"])
        exit(1)


def get_model_params(network_config):
    """Convert a configuration to actual model parameters

    Parameters
    ----------
    network_config : dict
        Dictionary containing the configuration options for the network.

    Returns
    -------
    model_params : dict
        Dictionary containing the actual parameters to be passed to the `net_*` functions in `models`.
    """
    model_params = {}
    if network_config["input_3x3"] and not network_config["arch"].startswith("wider"):
        model_params["input_3x3"] = True
    model_params["norm_act"] = _get_norm_act(network_config)
    model_params["classes"] = network_config["classes"]
    if not network_config["arch"].startswith("wider"):
        model_params["dilation"] = network_config["dilation"]
    return model_params
