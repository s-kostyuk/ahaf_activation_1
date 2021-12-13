import typing

import torch
import matplotlib.pyplot as plt

from libs import load_network, create_network


def get_random_idxs(max_i, cnt=10) -> typing.Sequence:
    return [int(torch.randint(size=(1,), low=0, high=max_i)) for _ in range(cnt)]


def random_selection(params, idxs):
    return [params[i] for i in idxs]


def visualize_neuron(gamma, beta, x, subfig):
    y = torch.sigmoid(gamma * x) * beta * x
    x_view = x.cpu().numpy()
    y_view = y.cpu().numpy()
    subfig.plot(x_view, y_view)


def visualize_activations(ahaf_params, fig, rows, show_subtitles=True):
    num_neurons = len(ahaf_params) // 2
    start_index = max(0, num_neurons - rows)
    cols = 5

    x = torch.arange(start=-10, end=4.0, step=0.1, device=ahaf_params[0].device)

    gs = plt.GridSpec(rows, cols)

    for i in range(rows):
        param_idx = start_index + i
        all_gamma = ahaf_params[param_idx * 2].view(-1)
        all_beta = ahaf_params[param_idx * 2 + 1].view(-1)
        sel = get_random_idxs(max_i=len(all_gamma), cnt=cols)
        sel_gamma = random_selection(all_gamma, sel)
        sel_beta = random_selection(all_beta, sel)

        for j in range(cols):
            subfig = fig.add_subplot(gs[i, j])
            if show_subtitles:
                subfig.set_title("L{} F{}".format(i, sel[j]))
            gamma = sel_gamma[j]
            beta = sel_beta[j]
            visualize_neuron(gamma, beta, x, subfig=subfig)


def load_and_visualize(
        net_name: str, dataset_name: str, act_name: str, batch_size: int, start_epoch: int, frozen_act: bool,
        trainer: typing.Optional[str], result_path, max_rows=None
):
    net = create_network(net_name, dataset_name, act_name, frozen_act)
    load_network(net, net_name, dataset_name, act_name, batch_size, start_epoch, frozen_act, trainer)

    ahaf_params = net.dsu2_param_sets[1]
    num_neurons = len(ahaf_params) // 2
    show_subtitles = False

    if max_rows is None:
        rows = num_neurons
    else:
        rows = min(max_rows, num_neurons)

    height = 1.17 * rows

    fig = plt.figure(tight_layout=True, figsize=(7, height))
    with torch.no_grad():
        visualize_activations(ahaf_params, fig, rows, show_subtitles)

    plt.savefig(result_path, dpi=300, format='svg')


def main():
    torch.manual_seed(seed=128)
    max_rows = 2
    load_and_visualize(
        net_name='LeNetAhaf', dataset_name='F-MNIST', act_name='SiL', frozen_act=False, trainer="dsu4",
        batch_size=64, start_epoch=100, result_path="runs/func_view_LeNet_F-MNIST_SiL.svg", max_rows=max_rows
    )
    load_and_visualize(
        net_name='LeNetAhaf', dataset_name='F-MNIST', act_name='ReLU', frozen_act=False, trainer="dsu4",
        batch_size=64, start_epoch=100, result_path="runs/func_view_LeNet_F-MNIST_ReLU.svg", max_rows=max_rows
    )
    load_and_visualize(
        net_name='KerasNetAhaf', dataset_name='CIFAR10', act_name='SiL', frozen_act=False, trainer="dsu4",
        batch_size=64, start_epoch=100, result_path="runs/func_view_KerasNet_CIFAR-10_SiL.svg", max_rows=max_rows
    )
    load_and_visualize(
        net_name='KerasNetAhaf', dataset_name='CIFAR10', act_name='ReLU', frozen_act=False, trainer="dsu4",
        batch_size=64, start_epoch=100, result_path="runs/func_view_KerasNet_CIFAR-10_ReLU.svg", max_rows=max_rows
    )


if __name__ == "__main__":
    main()
