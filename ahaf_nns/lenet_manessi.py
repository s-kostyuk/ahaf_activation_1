import torch.nn

from libs import activation_name_to_function


class LeNetManessi(torch.nn.Module):
    def __init__(self, *, flavor='MNIST', act='ReLU'):
        super(LeNetManessi, self).__init__()

        if flavor == 'MNIST' or flavor == 'F-MNIST':
            self._init_as_manessi_mnist()
        elif flavor == 'CIFAR10':
            self._init_as_manessi_cifar()
        else:
            raise NotImplemented("Other flavors of LeNet-5 are not supported")

        self._init_as_manessi_common(act=act)

    def _init_as_manessi_mnist(self):
        self._image_channels = 1
        self._fc3_in_features = 4 * 4 * 50

    def _init_as_manessi_cifar(self):
        self._image_channels = 3
        self._fc3_in_features = 5 * 5 * 50

    def _init_as_manessi_common(self, act: str):
        self._common_act = activation_name_to_function(act)

        # TODO: Check bias
        self.conv1 = torch.nn.Conv2d(
            in_channels=self._image_channels, out_channels=20, kernel_size=(5, 5),
            stride=(1, 1), padding=(0, 0), bias=False
        )
        self.act1 = self._common_act
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        # TODO: Check bias
        self.conv2 = torch.nn.Conv2d(
            in_channels=20, out_channels=50, kernel_size=(5, 5),
            stride=(1, 1), padding=(0, 0), bias=False
        )
        self.act2 = self._common_act
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.fc3 = torch.nn.Linear(
            in_features=self._fc3_in_features, out_features=500,
            bias=True
        )
        self.act3 = self._common_act

        self.fc4 = torch.nn.Linear(
            in_features=500, out_features=10,
            bias=False
        )
        #self.act4 = self._common_act

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        # dim 0 - mini-batch items
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.fc3(x)
        x = self.act3(x)

        x = self.fc4(x)
        #x = self.act4(x)

        return x
