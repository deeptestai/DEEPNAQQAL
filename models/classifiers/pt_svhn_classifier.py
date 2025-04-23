from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from keras.layers import Input
# from svhn.ModelA import ModelA
import numpy as np


class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SVHN_classifier(nn.Module):
    def __init__(self, num_classes=10):

        super(SVHN_classifier, self).__init__()
        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(5, 5),
                               padding='same', stride=1)
        self.relu1 = nn.ReLU()

        self.conv2 = Conv2dSame(96, 96, kernel_size=(3, 3), stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(96, 192, kernel_size=(5, 5), padding='same', stride=1)

        self.relu3 = nn.ReLU()

        self.conv4 = Conv2dSame(192, 192, kernel_size=(3, 3),
                               stride=2)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 192, kernel_size=(3, 3),
                               padding='same', stride=1)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(192, 192, kernel_size=(1, 1),
                               padding='valid', stride=1)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(192, self.num_classes, kernel_size=(1, 1),
                               padding='valid', stride=1)
        self.relu7 = nn.ReLU()
        
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        out = (self.relu1(self.conv1(x)))
        out = self.relu2(self.conv2(out))
        # expected (16,16,96)
        out = self.relu3(self.conv3(out))
        # expected (16,16,192)
        out = self.relu4(self.conv4(out))
        # expected (8,8,192)
        out = self.relu5(self.conv5(out))
        # expected (8,8,192)
        out = self.relu6(self.conv6(out))
        # expected (8,8,192)
        out = self.relu7(self.conv7(out))
        # expected (8,8,192)
         
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.num_classes)
        out = self.out(out)
        return out
    
    def reset_last_layer(self, num_classes=2):
        self.num_classes = num_classes
        self.conv7 = nn.Conv2d(192, self.num_classes, kernel_size=(1, 1),
                               padding='valid', stride=1)


# # input image dimensions
# img_rows, img_cols = 32, 32
# img_chn = 3
# img_dim = 1000
# input_shape = (img_rows, img_cols, img_chn)
# # define input tensor as a placeholder
# input_tensor = Input(shape=input_shape)
# # load multiple models sharing same input tensor
# model1 = ModelA(input_tensor)
# tf_weights = model1.get_weights()

# net = SVHN_classifier()
# sd = net.state_dict()

# # copy tf weights to pt
# def translate_convw(weights, index):
#     convw = weights[index]
#     convw = np.transpose(convw, (3, 2, 0, 1))
#     convw = torch.from_numpy(convw)
#     return convw

# def translate_bias(weights, index):
#     convb = weights[index]
#     convb = torch.from_numpy(convb)
#     return convb


# sd['conv1.weight'] = translate_convw(tf_weights, 0)
# sd['conv1.bias'] = translate_bias(tf_weights, 1)
# sd['conv2.weight'] = translate_convw(tf_weights, 2)
# sd['conv2.bias'] = translate_bias(tf_weights, 3)
# sd['conv3.weight'] = translate_convw(tf_weights, 4)
# sd['conv3.bias'] = translate_bias(tf_weights, 5)
# sd['conv4.weight'] = translate_convw(tf_weights, 6)
# sd['conv4.bias'] = translate_bias(tf_weights, 7)
# sd['conv5.weight'] = translate_convw(tf_weights, 8)
# sd['conv5.bias'] = translate_bias(tf_weights, 9)
# sd['conv6.weight'] = translate_convw(tf_weights, 10)
# sd['conv6.bias'] = translate_bias(tf_weights, 11)
# sd['conv7.weight'] = translate_convw(tf_weights, 12)
# sd['conv7.bias'] = translate_bias(tf_weights, 13)

# torch.save(sd, "svhn_class.pt")



