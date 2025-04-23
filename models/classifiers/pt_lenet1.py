from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
#from keras.layers import Input
#from mnist.Model1 import Model1
import torch.nn.functional as F


# pt model definition
class LeNet1(nn.Module):
    def __init__(self):
        kernel_size = (5, 5)
        super(LeNet1, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 4, kernel_size=kernel_size,
                               padding='same', stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)

        # Block 2
        self.conv2 = nn.Conv2d(4, 12, kernel_size=kernel_size,
                               padding='same', stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        # 7x7x12 = 588
        self.out = nn.Linear(588, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = torch.permute(out, (0, 2, 3, 1))
        out = torch.flatten(out, start_dim=1)
        #(10000, 588)
        out = self.out(out)  # [batch_size, 10]
        out = F.softmax(out, dim=1)
        return out


# # input image dimensions
# img_rows, img_cols = 28, 28
# img_dim = img_rows * img_cols
# input_shape = (img_rows, img_cols, 1)

# # load tf weights
# input_tensor = Input(shape=input_shape)
# model1 = Model1(input_tensor=input_tensor)
# tf_weights = model1.get_weights()

# # load pt state_dict
# net = LeNet1()
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

# out_w = tf_weights[4]
# out_w = np.transpose(out_w)
# out_w = torch.from_numpy(out_w)
# sd['out.weight'] = out_w

# sd['out.bias'] = translate_bias(tf_weights, 5)

# torch.save(sd, "lenet1.pt")
