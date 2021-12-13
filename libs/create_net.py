from ahaf_nns import LeNetManessi, LeNetAhaf, KerasNetManessi, KerasNetAhaf


def create_network(net_name: str, dataset_name: str, act_name: str, frozen_act: bool = True):
    if net_name == 'LeNet':
        net = LeNetManessi(flavor=dataset_name, act=act_name)
    elif net_name == 'KerasNet':
        net = KerasNetManessi(flavor=dataset_name, act=act_name)
    elif net_name == 'LeNetAhaf':
        net = LeNetAhaf(flavor=dataset_name, act_init_as=act_name, frozen_act=frozen_act)
    elif net_name == 'KerasNetAhaf':
        net = KerasNetAhaf(flavor=dataset_name, act_init_as=act_name, frozen_act=frozen_act)
    elif net_name == 'LeNetAhafMin':
        net = LeNetAhaf(flavor=dataset_name, act_init_as=act_name, frozen_act=frozen_act, enc_classic_act=act_name)
    elif net_name == 'KerasNetAhafMin':
        net = KerasNetAhaf(flavor=dataset_name, act_init_as=act_name, frozen_act=frozen_act, enc_classic_act=act_name)
    else:
        raise NotImplemented("Networks other than LeNet-5 and KerasNet are not supported")

    return net
