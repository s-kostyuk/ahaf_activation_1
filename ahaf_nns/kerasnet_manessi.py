import torch.nn

from libs import activation_name_to_function


class KerasNetManessi(torch.nn.Module):
    """
    KerasNet - CNN implementation evaluated in arXiv 1801.09403.
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
    def __init__(self, *, flavor='MNIST', act='ReLU'):
        super(KerasNetManessi, self).__init__()

        if flavor == 'MNIST' or flavor == 'F-MNIST':
            self._init_as_manessi_mnist()
        elif flavor == 'CIFAR10':
            self._init_as_manessi_cifar()
        else:
            raise NotImplemented("Other flavors of KerasNet are not supported")

        self._init_as_manessi_common(act=act)

    def _init_as_manessi_mnist(self):
        self._image_channels = 1
        self._fc7_in_features = 5 * 5 * 64

    def _init_as_manessi_cifar(self):
        self._image_channels = 3
        self._fc7_in_features = 6 * 6 * 64

    def _init_as_manessi_common(self, act: str):
        self._common_act = activation_name_to_function(act)

        self.conv1 = torch.nn.Conv2d(
            in_channels=self._image_channels, out_channels=32, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=True
        )
        self.act1 = self._common_act

        self.conv2 = torch.nn.Conv2d(
            in_channels=self.conv1.out_channels, out_channels=32, kernel_size=(3, 3),
            stride=(1, 1), padding=(0, 0), bias=True
        )
        self.act2 = self._common_act

        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop3 = torch.nn.Dropout2d(p=0.25)

        self.conv4 = torch.nn.Conv2d(
            in_channels=self.conv2.out_channels, out_channels=64, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=True
        )
        self.act4 = self._common_act

        self.conv5 = torch.nn.Conv2d(
            in_channels=self.conv4.out_channels, out_channels=64, kernel_size=(3, 3),
            stride=(1, 1), padding=(0, 0), bias=True
        )
        self.act5 = self._common_act

        self.pool6 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop6 = torch.nn.Dropout2d(p=0.25)

        self.fc7 = torch.nn.Linear(
            in_features=self._fc7_in_features, out_features=512, bias=True
        )
        self.act7 = self._common_act
        self.drop7 = torch.nn.Dropout2d(p=0.5)

        self.fc8 = torch.nn.Linear(
            in_features=self.fc7.out_features, out_features=10, bias=True
        )

        # softmax is embedded in pytorch's loss function

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool6(x)
        x = self.drop6(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.fc7(x)
        x = self.act7(x)
        x = self.drop7(x)

        x = self.fc8(x)

        return x
