import typing

import torch


RESULTS_PATH = "runs"


def save_network(
        net: torch.nn.Module, net_name: str, dataset_name: str,
        act_name: str, batch_size: int, epochs: int, frozen: bool, trainer: typing.Optional[str]
):
    frozen_str = "" if frozen else "_ul"
    trainer_str = "" if trainer is None else "_{}".format(trainer)
    path = "{}/net_{}{}_{}_{}_bs{}{}_ep{}.bin".format(
        RESULTS_PATH, net_name, frozen_str, dataset_name, act_name, batch_size, trainer_str, epochs
    )
    torch.save(net.state_dict(), path)


def save_optimizer(
        opt: torch.optim.Optimizer, net_name: str, dataset_name: str, act_name: str, batch_size: int, epochs: int
):
    path = "{}/opt_{}_{}_{}_bs{}_ep{}.bin".format(
        RESULTS_PATH, net_name, dataset_name, act_name, batch_size, epochs
    )
    torch.save(opt.state_dict(), path)


def load_network(
        net: torch.nn.Module, net_name: str, dataset_name: str,
        act_name: str, batch_size: int, start_epoch: int, frozen: bool, trainer: typing.Optional[str]
):
    frozen_str = "" if frozen else "_ul"
    trainer_str = "" if trainer is None else "_{}".format(trainer)
    path = "{}/net_{}{}_{}_{}_bs{}{}_ep{}.bin".format(
        RESULTS_PATH, net_name, frozen_str, dataset_name, act_name, batch_size, trainer_str, start_epoch
    )
    net.load_state_dict(torch.load(path))


def load_optimizer(
        opt: torch.optim.Optimizer, net_name: str, dataset_name: str, act_name: str, batch_size: int, start_epoch: int
):
    path = "{}/opt_{}_{}_{}_bs{}_ep{}.bin".format(
        RESULTS_PATH, net_name, dataset_name, act_name, batch_size, start_epoch
    )
    opt.load_state_dict(torch.load(path))
