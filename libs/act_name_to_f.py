from torch.functional import F


def activation_name_to_function(name: str):
    if name == 'ReLU':
        return F.relu
    elif name == 'tanh':
        return F.tanh
    elif name == 'id':
        return lambda x: x
    elif name == 'SiL':
        return F.silu
    else:
        raise NotImplemented("Other functions than ReLU, tanh and id are not supported")
