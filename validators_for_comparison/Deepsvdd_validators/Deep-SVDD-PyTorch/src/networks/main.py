from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .svhn_LeNet import svhn_LeNet, svhn_LeNet_Autoencoder
from .svhn_LeNet_elu import svhn_LeNet_ELU, svhn_LeNet_ELU_Autoencoder
from .img_resnet50 import ResNet50Autoencoder

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'svhn_LeNet', 'svhn_LeNet_elu','img_resnet50')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'svhn_LeNet':
        net = svhn_LeNet()

    if net_name == 'svhn_LeNet_elu':
        net = svhn_LeNet_ELU()

    if net_name == 'img_resnet50':
        net = ResNet50Autoencoder()


    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'svhn_LeNet', 'svhn_LeNet_elu','img_resnet50')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'svhn_LeNet':
        ae_net = svhn_LeNet_Autoencoder()

    if net_name == 'svhn_LeNet_elu':
        ae_net = svhn_LeNet_ELU_Autoencoder()

    if net_name == 'img_resnet50':
        ae_net = ResNet50Autoencoder()

    return ae_net
