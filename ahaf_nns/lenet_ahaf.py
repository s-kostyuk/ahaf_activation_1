from typing import Iterator, Sequence

import torch.nn

from .ahaf import AHAF
from libs import activation_name_to_function


class LeNetAhaf(torch.nn.Module):
    def __init__(self, *, flavor='MNIST', act_init_as='ReLU', frozen_act=True, enc_classic_act=None):
        super(LeNetAhaf, self).__init__()

        if flavor == 'MNIST' or flavor == 'F-MNIST':
            self._init_as_ahaf_mnist()
        elif flavor == 'CIFAR10':
            self._init_as_ahaf_cifar()
        else:
            raise NotImplemented("Other flavors of LeNet-5 are not supported")

        if enc_classic_act is None:
            self._common_enc_act = None
        else:
            self._common_enc_act = activation_name_to_function(act_init_as)

        self._init_as_ahaf_common(act_init_as=act_init_as)

        if frozen_act:
            self.freeze_act()

    def _init_as_ahaf_mnist(self):
        self._image_channels = 1
        self._fc3_in_features = 4 * 4 * 50
        self._act1_img_dims = (24, 24)
        self._act2_img_dims = (8, 8)

    def _init_as_ahaf_cifar(self):
        self._image_channels = 3
        self._fc3_in_features = 5 * 5 * 50
        self._act1_img_dims = (28, 28)
        self._act2_img_dims = (10, 10)

    def _init_as_ahaf_common(self, act_init_as: str):
        # TODO: Check bias
        self.conv1 = torch.nn.Conv2d(
            in_channels=self._image_channels, out_channels=20, kernel_size=(5, 5),
            stride=(1, 1), padding=(0, 0), bias=False
        )
        if self._common_enc_act is None:
            self.act1 = AHAF(
                size=(self.conv1.out_channels, *self._act1_img_dims),
                init_as=act_init_as
            )
        else:
            self.act1 = self._common_enc_act

        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        # TODO: Check bias
        self.conv2 = torch.nn.Conv2d(
            in_channels=20, out_channels=50, kernel_size=(5, 5),
            stride=(1, 1), padding=(0, 0), bias=False
        )
        if self._common_enc_act is None:
            self.act2 = AHAF(
                size=(self.conv2.out_channels, *self._act2_img_dims),
                init_as=act_init_as
            )
        else:
            self.act2 = self._common_enc_act

        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self._flatter = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.fc3 = torch.nn.Linear(
            in_features=self._fc3_in_features, out_features=500,
            bias=True
        )
        self.act3 = AHAF(
            size=(self.fc3.out_features,),
            init_as=act_init_as
        )

        self.fc4 = torch.nn.Linear(
            in_features=500, out_features=10,
            bias=False
        )

        self._sequence = [
            self.conv1, self.act1, self.pool1,
            self.conv2, self.act2, self.pool2,
            self._flatter,
            self.fc3, self.act3,
            self.fc4
        ]

    def forward(self, x):
        for mod in self._sequence:
            x = mod(x)

        return x

    @property
    def encoder_act_params(self) -> Iterator[torch.nn.Parameter]:
        return [*self.act1.parameters(), *self.act2.parameters()]

    @property
    def act_params(self) -> Iterator[torch.nn.Parameter]:
        result = []

        if self._common_enc_act is None:
            result.extend(self.encoder_act_params)

        result.extend(self.act3.parameters())
        return result

    def freeze_act(self):
        for param in self.act_params:
            param.requires_grad = False

    def unfreeze_act(self):
        for param in self.act_params:
            param.requires_grad = True

    @property
    def dsu_param_sets(self) -> Sequence[Sequence[torch.nn.Parameter]]:
        result = []
        batch = []

        for mod in self._sequence:
            if not isinstance(mod, torch.nn.Module):
                continue

            if not isinstance(mod, AHAF):
                batch.extend(mod.parameters())
                continue

            # AHAF function reached - save what we've got
            if batch:
                result.append(batch)

            # save AHAF params
            batch = [*mod.parameters()]
            result.append(batch)

            # reset the batch of parameters
            batch = []

        if batch:
            result.append(batch)

        return result

    @property
    def dsu2_param_sets(self) -> Sequence[Sequence[torch.nn.Parameter]]:
        ahaf_params = []
        non_ahaf_params = []

        for mod in self._sequence:
            if not isinstance(mod, torch.nn.Module):
                continue

            if isinstance(mod, AHAF):
                ahaf_params.extend(mod.parameters())
            else:
                non_ahaf_params.extend(mod.parameters())

        return [non_ahaf_params, ahaf_params]
