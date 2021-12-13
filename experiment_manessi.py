import csv
import typing
import os
import warnings

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchinfo
import torchvision.transforms
import torchvision.datasets

from libs.running_stat import RunningStat
from libs.multiopt_proxy import MultiOptProxy
from libs import create_network
from libs import save_network, load_network, save_optimizer, load_optimizer

DynamicData = typing.List[typing.Dict[str, typing.Any]]


DEBUG = False
DEBUG_DSU = True
RESULTS_PATH = "runs"
SAVE_MODEL_ENABLED = True
SAVE_OPT_ENABLED = False
SAVE_DYNAMICS_ENABLED = True
LOAD_MODEL_ENABLED = False
TRAIN_CLASSIC = False
TRAIN_AHAF = True


def create_results_folder():
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)


def save_dynamic_data(
        dynamic_data: DynamicData, net_name: str, dataset_name: str,
        act_name: str, batch_size: int, epochs: int, frozen: bool, trainer: typing.Optional[str]
):
    frozen_str = "" if frozen else "_ul"
    trainer_str = "" if trainer is None else "_{}".format(trainer)
    path = "{}/dynamics_{}{}_{}_{}_bs{}{}_ep{}.csv".format(
        RESULTS_PATH, net_name, frozen_str, dataset_name, act_name, batch_size, trainer_str, epochs
    )
    fields = "epoch", "train_loss_mean", "train_loss_var", "test_acc", "lr"

    with open(path, mode='w') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(dynamic_data)


def params_to_icons(params: typing.Iterator[torch.nn.Parameter]) -> typing.Iterator[str]:
    return ['🟢' if p.requires_grad else '🔴' for p in params]


def print_param_view_to_file(net, net_name, dataset_name, frozen):
    def trainable_params():
        for p in net.parameters():
            if p.requires_grad:
                yield str(p)

    frozen_str = "" if frozen else "_ul"
    file_path = "{}/{}{}_{}_param_view.txt".format(RESULTS_PATH, frozen_str, net_name, dataset_name)

    with open(file_path, mode="w") as f:
        f.writelines(trainable_params())


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using GPU computing unit")
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print("Cuda computing capability: {}.{}".format(*torch.cuda.get_device_capability(device)))
    else:
        print("Using CPU computing unit")
        device = torch.device('cpu')

    return device


def get_mnist_dataset(augment: bool = False) -> typing.Tuple[torch.utils.data.Dataset, ...]:
    if augment:
        augments = (
            # as in Keras - each second image is flipped
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # assuming that the values from git.io/JuHV0 were used in arXiv 1801.09403
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        )
    else:
        augments = ()

    train_set = torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(), *augments)
        )
    )

    test_set = torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(),)
        )
    )

    return train_set, test_set


def get_cifar10_dataset(augment: bool = False) -> typing.Tuple[torch.utils.data.Dataset, ...]:
    if augment:
        augments = (
            # as in Keras - each second image is flipped
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # assuming that the values from git.io/JuHV0 were used in arXiv 1801.09403
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        )
    else:
        augments = ()

    train_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(), *augments)
        )
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data/cifar",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(),)
        )
    )

    return train_set, test_set


def train_non_dsu(batches, dev, net, error_fn, opt):
    loss_stat = RunningStat()

    for mb in batches:
        # Get output
        x, y = mb[0].to(dev), mb[1].to(dev)

        y_hat = net.forward(x)
        loss = error_fn(y_hat, target=y)
        loss_stat.push(loss.item())

        # Update parameters
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss_stat


def train_dsu(batches, dev, net, error_fn, multi_opt, trainer="dsu"):
    loss_stat = None

    prev_ps = None
    for p in net.parameters():
        p.requires_grad = False

    # Make sure that the data is not changed during double-stage parameters update.
    # In particular - prevent the augmented data from changing here.
    mb_cache = list(batches)

    if trainer == "dsu2" or trainer == "dsu4":
        params_sets = net.dsu2_param_sets
    else:
        params_sets = net.dsu_param_sets

    i = 0

    for ps in reversed(params_sets):
        opt = multi_opt.get_opt(i)
        i += 1

        if prev_ps is not None:
            for p in prev_ps:
                p.requires_grad = False

        for p in ps:
            p.requires_grad = True

        prev_ps = ps

        if DEBUG_DSU:
            print(params_to_icons(net.parameters()))

        loss_stat = RunningStat()

        for mb in mb_cache:
            x, y = mb[0].to(dev), mb[1].to(dev)
            y_hat = net.forward(x)
            loss = error_fn(y_hat, target=y)
            loss_stat.push(loss.item())

            # Update parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

        if DEBUG_DSU:
            print("Loss for this parameter set: {}, variance: {}".format(loss_stat.mean, loss_stat.variance))

    return loss_stat


def train_eval(
        net_name: str, dataset_name: str, act_name: str, frozen_act: bool = True, trainer: typing.Optional[str] = None
) -> None:
    """
    Dataset A:
    - FashionMNIST
    - 50 000 images - train set
    - 10 000 images - eval set

    Dataset B:
    - CIFAR-10
    - 50 000 images - train set
    - 10 000 images - eval set

    Preprocessing (datasets A and B):
    - divide pixels by 255 (pre-done in the torchvision's dataset)
    - augment: random horizontal flip and image shifting

    Training:
    - RMSprop
    - lr = 10**-4
    - lr_decay_mb = 10**-6
    - batch_size = ???

    :return: None
    """
    batch_size = 64
    nb_start_ep = 0
    nb_epochs = 100
    rand_seed = 42

    if SAVE_MODEL_ENABLED or SAVE_DYNAMICS_ENABLED or SAVE_OPT_ENABLED:
        create_results_folder()

    dynamic_data = []  # type: typing.List[typing.Dict]

    print("\nTraining {} network with {} activation on {} dataset with batch size {}".format(
        net_name, act_name, dataset_name, batch_size
    ))

    dev = get_device()
    torch.manual_seed(rand_seed)

    if dataset_name == 'F-MNIST':
        train_set, test_set = get_mnist_dataset(augment=True)
        input_size = (batch_size, 1, 28, 28)
    elif dataset_name == 'CIFAR10':
        train_set, test_set = get_cifar10_dataset(augment=True)
        input_size = (batch_size, 3, 32, 32)
    else:
        raise NotImplemented("Datasets other than Fashion-MNIST and CIFAR-10 are not supported")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1000, num_workers=4
    )

    net = create_network(net_name, dataset_name, act_name, frozen_act)

    if LOAD_MODEL_ENABLED:
        load_network(net, net_name, dataset_name, act_name, batch_size, nb_start_ep, frozen_act, trainer)

    net.to(device=dev)

    torchinfo.summary(net, input_size=input_size)
    # print(net.dsu_param_sets)

    if DEBUG:
        print(params_to_icons(net.parameters()))
        print_param_view_to_file(net, net_name, dataset_name, frozen_act)
        return

    error_fn = torch.nn.CrossEntropyLoss()

    def build_opt(params):
        return torch.optim.RMSprop(
            params=params,
            lr=1e-4,
            alpha=0.9,  # default Keras
            momentum=0.0,  # default Keras
            eps=1e-7,  # default Keras
            centered=False  # default Keras
        )

    def inv_time_decay(step: int) -> float:
        """
        InverseTimeDecay in Keras, default decay formula used in keras.optimizers.Optimizer
        as per OptimizerV2._decayed_lr() - see https://git.io/JEKA6 and https://git.io/JEKx2.
        """
        decay_steps = 1  # update after each epoch
        decay_rate = 1.562e-3

        return 1.0 / (1.0 + decay_rate * step / decay_steps)

    def build_sched(optimizer):
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=inv_time_decay
        )

    multiopt = MultiOptProxy(build_opt, build_sched, net, trainer)

    for epoch in range(nb_start_ep, nb_epochs):
        net.train()

        if trainer is not None:
            loss_stat = train_dsu(train_loader, dev, net, error_fn, multiopt, trainer=trainer)
        else:
            opt = multiopt.get_opt(0)
            loss_stat = train_non_dsu(train_loader, dev, net, error_fn, opt)

        net.eval()

        with torch.no_grad():
            net.eval()
            test_total = 0
            test_correct = 0

            for batch in test_loader:
                x = batch[0].to(dev)
                y = batch[1].to(dev)
                y_hat = net(x)
                _, pred = torch.max(y_hat.data, 1)
                test_total += y.size(0)
                test_correct += (pred == y).sum().item()

            net.train()

            print("Train set loss stat: m={}, var={}".format(loss_stat.mean, loss_stat.variance))
            print("Epoch: {}. Test set accuracy: {:.2%}".format(epoch, test_correct / test_total))
            print("Current LR: {}".format(multiopt.get_last_lr()))

            if SAVE_DYNAMICS_ENABLED:
                dynamic_data.append({
                    "train_loss_mean": loss_stat.mean,
                    "train_loss_var": loss_stat.variance,
                    "test_acc": test_correct / test_total,
                    "lr": multiopt.get_last_lr()[0],
                    "epoch": epoch
                })

        # Classic - update LR on each epoch
        multiopt.sched_step()

    if SAVE_MODEL_ENABLED:
        save_network(net, net_name, dataset_name, act_name, batch_size, nb_epochs, frozen_act, trainer)

    # TODO: Fix saving of opts
    if SAVE_OPT_ENABLED:
        warnings.warn("Saving optimizer parameters is not supported for MultioptProxy")
        #save_optimizer(opt, net_name, dataset_name, act_name, batch_size, nb_epochs)

    if SAVE_DYNAMICS_ENABLED:
        save_dynamic_data(dynamic_data, net_name, dataset_name, act_name, batch_size, nb_epochs, frozen_act, trainer)


def main():
    if TRAIN_CLASSIC:
        act_name = 'SiL'
        train_eval(net_name='LeNet', dataset_name='F-MNIST', act_name=act_name)
        train_eval(net_name='LeNet', dataset_name='CIFAR10', act_name=act_name)
        train_eval(net_name='KerasNet', dataset_name='F-MNIST', act_name=act_name)
        train_eval(net_name='KerasNet', dataset_name='CIFAR10', act_name=act_name)

    if TRAIN_AHAF:
        frozen_act = False
        act_name = 'SiL'
        trainer = None  # double-stage update of parameters
        train_eval(net_name='LeNetAhaf', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        train_eval(net_name='LeNetAhaf', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        train_eval(net_name='KerasNetAhaf', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        train_eval(net_name='KerasNetAhaf', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)

        act_name = 'ReLU'
        #train_eval(net_name='LeNetAhaf', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='LeNetAhaf', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='KerasNetAhaf', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='KerasNetAhaf', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)

        trainer = "dsu4"
        act_name = 'SiL'
        #train_eval(net_name='LeNetAhaf', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='LeNetAhaf', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='KerasNetAhaf', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='KerasNetAhaf', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)

        act_name = 'ReLU'
        #train_eval(net_name='LeNetAhaf', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='LeNetAhaf', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='KerasNetAhaf', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='KerasNetAhaf', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)

        #train_eval(net_name='LeNetAhafMin', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='LeNetAhafMin', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='KerasNetAhafMin', dataset_name='F-MNIST', act_name=act_name, frozen_act=frozen_act, trainer=trainer)
        #train_eval(net_name='KerasNetAhafMin', dataset_name='CIFAR10', act_name=act_name, frozen_act=frozen_act, trainer=trainer)


if __name__ == "__main__":
    main()
