#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from typing import Sequence, Dict, Tuple, Iterable, List


def tuple_to_file_name(options: Sequence[str]) -> str:
    net = options[0]
    var = options[1]
    trainer = options[2]
    activation = options[3]
    bs = options[4]
    dataset = options[5]

    net_str = "{}{}_ul".format(net, var) if var else net
    trainer_str = "_" + trainer if trainer else ""

    return "dynamics_{}_{}_{}_bs{}{}_ep{}.csv".format(
        net_str, dataset, activation, bs, trainer_str, 100
    )


def load_results(file_path) -> Iterable[Dict]:
    # fields = ("epoch", "train_loss_mean", "train_loss_var", "test_acc", "lr")
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_base_legend(options):
    var = options[1]
    trainer = options[2]
    activation = options[3]

    trainer_str = "DSPT" if isinstance(trainer, str) and "dsu" in trainer else "no DSPT"

    if var == "Ahaf":
        return "{}-like AHAF, {}".format(activation, trainer_str)
    else:
        return "{} activation, no DSPT".format(activation)


def analyze_net(options):
    file_path = "runs/{}".format(tuple_to_file_name(options))
    results = load_results(file_path)
    base_legend = to_base_legend(options)

    acc = []
    loss = []

    for r in results:
        acc.append(float(r["test_acc"]) * 100.0)
        loss.append(float(r["train_loss_mean"]))

    return base_legend, acc, loss


def plot_networks(fig, *nets):
    legends1 = []
    legends2 = []

    gs = plt.GridSpec(1, 2)

    acc_fig = fig.add_subplot(gs[0, 0])
    #acc_loc = plticker.LinearLocator(numticks=10)
    #acc_fig.yaxis.set_major_locator(acc_loc)
    acc_fig.set_xlabel('epoch')
    acc_fig.set_ylabel('test accuracy, %')

    loss_fig = fig.add_subplot(gs[0, 1])
    #loss_loc = plticker.LinearLocator(numticks=10)
    #loss_fig.yaxis.set_major_locator(loss_loc)
    loss_fig.set_xlabel('epoch')
    loss_fig.set_ylabel('training loss')

    for net in nets:
        try:
            base_legend, acc, loss = analyze_net(net)
        except Exception as e:
            print("Exception: {}, skipped".format(e))
            continue

        x = tuple(range(len(acc)))

        legends1.append(
            base_legend
        )
        legends2.append(
            base_legend
        )
        acc_fig.plot(x, acc)
        loss_fig.plot(x, loss)

    acc_fig.legend(legends1)
    loss_fig.legend(legends2)


def visualize(nets, img_path_template: str, dataset: str, base_title=None):
    options = []
    for net in nets:
        options.append(
            (*net, dataset)
        )

    fig = plt.figure(tight_layout=True, figsize=(7, 3.5))
    if base_title is not None:
        title = "{}, test accuracy and training loss".format(base_title)
        fig.suptitle(title)

    plot_networks(fig, *options)

    #plt.show()
    plt.savefig(img_path_template.format(dataset), dpi=300, format='svg')


def main():
    batch_size = 64

    img_path_template_lenet = "runs/dynamics-lenet-{}_comparison.svg"
    nets_le_net = (
        ("LeNet", "", "", "ReLU", batch_size),
        ("LeNet", "", "", "SiL", batch_size),
        ("LeNet", "Ahaf", "", "ReLU", batch_size),
        ("LeNet", "Ahaf", "", "SiL", batch_size),
        ("LeNet", "Ahaf", "dsu4", "ReLU", batch_size),
        ("LeNet", "Ahaf", "dsu4", "SiL", batch_size),
        ("LeNet", "AhafMin", "dsu4", "ReLU", batch_size),
        ("LeNet", "AhafMin", "dsu4", "SiL", batch_size),
    )

    visualize(nets_le_net, img_path_template_lenet, "F-MNIST", "LeNet-5 on Fashion-MNIST")
    visualize(nets_le_net, img_path_template_lenet, "CIFAR10", "LeNet-5 on CIFAR-10")

    img_path_template_keras_net = "runs/dynamics-kerasnet-{}_comparison.svg"
    nets_keras_net = (
        ("KerasNet", "", "", "ReLU", batch_size),
        ("KerasNet", "", "", "SiL", batch_size),
        ("KerasNet", "Ahaf", "", "ReLU", batch_size),
        ("KerasNet", "Ahaf", "", "SiL", batch_size),
        ("KerasNet", "Ahaf", "dsu4", "ReLU", batch_size),
        ("KerasNet", "Ahaf", "dsu4", "SiL", batch_size),
        ("KerasNet", "AhafMin", "dsu4", "ReLU", batch_size),
        ("KerasNet", "AhafMin", "dsu4", "SiL", batch_size),
    )

    #visualize(nets_keras_net, img_path_template_keras_net, "F-MNIST", base_title="KerasNet on Fashion-MNIST")
    #visualize(nets_keras_net, img_path_template_keras_net, "CIFAR10", base_title="KerasNet on CIFAR-10")
    visualize(nets_keras_net, img_path_template_keras_net, "F-MNIST", base_title=None)
    visualize(nets_keras_net, img_path_template_keras_net, "CIFAR10", base_title=None)


if __name__ == "__main__":
    main()
