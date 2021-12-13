from typing import Iterator, Sequence

import torch.nn

from .ahaf import AHAF
from libs import activation_name_to_function


class KerasNetAhaf(torch.nn.Module):
    """
    KerasNet - CNN implementation evaluated in arXiv 1801.09403 **but** wih AHAF activations.
    The model is based on the example CNN implementation from Keras 1.x: git.io/JuHV0.

    Architecture:

    - 2D convolution 32 x (3,3) with (1,1) padding
    - activation
    - 2D convolution 32 x (3,3) w/o padding
    - activation
    - max pooling (2,2)
    - dropout 25%
    - 2D convolution 64 x (3,3) with (1,1) padding
    - activation
    - 2D convolution 64 x (3,3) w/o padding
    - activation
    - max pooling (2,2)
    - dropout 25%
    - fully connected, out_features = 512
    - activation
    - dropout 50%
    - fully connected, out_features = 10
    - softmax activation

    """
    def __init__(self, *, flavor='MNIST', act_init_as='ReLU', frozen_act=True, enc_classic_act=None):
        super(KerasNetAhaf, self).__init__()

        if flavor == 'MNIST' or flavor == 'F-MNIST':
            self._init_as_ahaf_mnist()
        elif flavor == 'CIFAR10':
            self._init_as_ahaf_cifar()
        else:
            raise NotImplemented("Other flavors of KerasNet are not supported")

        if enc_classic_act is None:
            self._common_enc_act = None
        else:
            self._common_enc_act = activation_name_to_function(act_init_as)

        self._init_as_ahaf_common(act_init_as=act_init_as)

        if frozen_act:
            self.freeze_act()

    def _init_as_ahaf_mnist(self):
        self._image_channels = 1
        self._fc7_in_features = 5 * 5 * 64
        self._act1_img_dims = (28, 28)
        self._act2_img_dims = (26, 26)
        self._act4_img_dims = (13, 13)
        self._act5_img_dims = (11, 11)

    def _init_as_ahaf_cifar(self):
        self._image_channels = 3
        self._fc7_in_features = 6 * 6 * 64
        self._act1_img_dims = (32, 32)
        self._act2_img_dims = (30, 30)
        self._act4_img_dims = (15, 15)
        self._act5_img_dims = (13, 13)

    def _init_as_ahaf_common(self, act_init_as: str):
        self.conv1 = torch.nn.Conv2d(
            in_channels=self._image_channels, out_channels=32, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=True
        )
        if self._common_enc_act is None:
            self.act1 = AHAF(
                size=(self.conv1.out_channels, *self._act1_img_dims),
                init_as=act_init_as
            )
        else:
            self.act1 = self._common_enc_act

        self.conv2 = torch.nn.Conv2d(
            in_channels=self.conv1.out_channels, out_channels=32, kernel_size=(3, 3),
            stride=(1, 1), padding=(0, 0), bias=True
        )
        if self._common_enc_act is None:
            self.act2 = AHAF(
                size=(self.conv2.out_channels, *self._act2_img_dims),
                init_as=act_init_as
            )
        else:
            self.act2 = self._common_enc_act

        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop3 = torch.nn.Dropout2d(p=0.25)

        self.conv4 = torch.nn.Conv2d(
            in_channels=self.conv2.out_channels, out_channels=64, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=True
        )
        if self._common_enc_act is None:
            self.act4 = AHAF(
                size=(self.conv4.out_channels, *self._act4_img_dims),
                init_as=act_init_as
            )
        else:
            self.act4 = self._common_enc_act

        self.conv5 = torch.nn.Conv2d(
            in_channels=self.conv4.out_channels, out_channels=64, kernel_size=(3, 3),
            stride=(1, 1), padding=(0, 0), bias=True
        )
        if self._common_enc_act is None:
            self.act5 = AHAF(
                size=(self.conv5.out_channels, *self._act5_img_dims),
                init_as=act_init_as
            )
        else:
            self.act5 = self._common_enc_act

        self.pool6 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop6 = torch.nn.Dropout2d(p=0.25)

        self._flatter = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.fc7 = torch.nn.Linear(
            in_features=self._fc7_in_features, out_features=512, bias=True
        )
        self.act7 = AHAF(
            size=(self.fc7.out_features,),
            init_as=act_init_as
        )
        self.drop7 = torch.nn.Dropout2d(p=0.5)

        self.fc8 = torch.nn.Linear(
            in_features=self.fc7.out_features, out_features=10, bias=True
        )

        # softmax is embedded in pytorch's loss function

        self._sequence = [
            self.conv1, self.act1, self.conv2, self.act2, self.pool3, self.drop3,
            self.conv4, self.act4, self.conv5, self.act5, self.pool6, self.drop6,
            self._flatter,
            self.fc7, self.act7, self.drop7,
            self.fc8
        ]

    def forward(self, x):
        for mod in self._sequence:
            x = mod(x)

        return x

    @property
    def encoder_act_params(self) -> Iterator[torch.nn.Parameter]:
        return [
            *self.act1.parameters(), *self.act2.parameters(),
            *self.act4.parameters(), *self.act5.parameters()
        ]

    @property
    def act_params(self) -> Iterator[torch.nn.Parameter]:
        result = []

        if self._common_enc_act is None:
            result.extend(self.encoder_act_params)

        result.extend(self.act7.parameters())
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
