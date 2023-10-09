""" Parts of the network model """

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D Conv
def conv1(in_planes, out_planes, stride=1):
	return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv2(in_planes, out_planes, stride=1):
	return nn.Conv1d(in_planes, out_planes, kernel_size=2, stride=stride, padding=0, bias=False)

def conv3(in_planes, out_planes, stride=1):
	return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv4(in_planes, out_planes, stride=1):
	return nn.Conv1d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False)

# 2D Conv
def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv2x2(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv4x4(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes,
			kernel_size=4, stride=stride, padding=1, 
			bias=False)

# 3D Conv
def conv1x1x1(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes, out_planes,
			kernel_size=1, stride=stride, padding=0, 
			bias=False)

def conv3x3x3(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes, out_planes,
			kernel_size=3, stride=stride, padding=1, 
			bias=False)

def conv4x4x4(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes, out_planes,
			kernel_size=4, stride=stride, padding=1, 
			bias=False)

# 1D Deconv
def deconv1(in_planes, out_planes, stride):
	return nn.ConvTranspose1d(in_planes, out_planes, 
			kernel_size=1, stride=stride, padding=0, output_padding=0, 
			bias=False)

def deconv2(in_planes, out_planes, stride):
	return nn.ConvTranspose1d(in_planes, out_planes, 
			kernel_size=2, stride=stride, padding=0, output_padding=0, 
			bias=False)

def deconv3(in_planes, out_planes, stride):
	return nn.ConvTranspose1d(in_planes, out_planes, 
			kernel_size=3, stride=stride, padding=1, output_padding=0, 
			bias=False)

def deconv4(in_planes, out_planes, stride):
	return nn.ConvTranspose1d(in_planes, out_planes, 
			kernel_size=4, stride=stride, padding=1, output_padding=0, 
			bias=False)

# 2D Deconv
def deconv1x1(in_planes, out_planes, stride):
	return nn.ConvTranspose2d(in_planes, out_planes, 
			kernel_size=1, stride=stride, padding=0, output_padding=0, 
			bias=False)

def deconv2x2(in_planes, out_planes, stride):
	return nn.ConvTranspose2d(in_planes, out_planes, 
			kernel_size=2, stride=stride, padding=0, output_padding=0, 
			bias=False)

def deconv3x3(in_planes, out_planes, stride):
	return nn.ConvTranspose2d(in_planes, out_planes, 
			kernel_size=3, stride=stride, padding=1, output_padding=0, 
			bias=False)

def deconv4x4(in_planes, out_planes, stride):
	return nn.ConvTranspose2d(in_planes, out_planes, 
			kernel_size=4, stride=stride, padding=1, output_padding=0, 
			bias=False)

# 3D Deconv
def deconv1x1x1(in_planes, out_planes, stride):
	return nn.ConvTranspose3d(in_planes, out_planes, 
			kernel_size=1, stride=stride, padding=0, output_padding=0, 
			bias=False)

def deconv3x3x3(in_planes, out_planes, stride):
	return nn.ConvTranspose3d(in_planes, out_planes, 
			kernel_size=3, stride=stride, padding=1, output_padding=0, 
			bias=False)

def deconv4x4x4(in_planes, out_planes, stride):
	return nn.ConvTranspose3d(in_planes, out_planes, 
			kernel_size=4, stride=stride, padding=1, output_padding=0, 
			bias=False)

def _make_layers(in_channels, output_channels, type, batch_norm=False, activation=None):
	layers = []

	if type == 'conv1_s1':
		layers.append(conv1(in_channels, output_channels, stride=1))
	elif type == 'conv2_s2':
		layers.append(conv2(in_channels, output_channels, stride=2))
	elif type == 'conv3_s1':
		layers.append(conv3(in_channels, output_channels, stride=1))
	elif type == 'conv4_s2':
		layers.append(conv4(in_channels, output_channels, stride=2))
	elif type == 'conv1x1_s1':
		layers.append(conv1x1(in_channels, output_channels, stride=1))
	elif type == 'conv2x2_s2':
		layers.append(conv2x2(in_channels, output_channels, stride=2))
	elif type == 'conv3x3_s1':
		layers.append(conv3x3(in_channels, output_channels, stride=1))
	elif type == 'conv4x4_s2':
		layers.append(conv4x4(in_channels, output_channels, stride=2))
	elif type == 'conv1x1x1_s1':
		layers.append(conv1x1x1(in_channels, output_channels, stride=1))
	elif type == 'conv3x3x3_s1':
		layers.append(conv3x3x3(in_channels, output_channels, stride=1))
	elif type == 'deconv3_s1':
		layers.append(deconv3(in_channels, output_channels, stride=1))
	elif type == 'deconv4_s2':
		layers.append(deconv4(in_channels, output_channels, stride=2))
	elif type == 'deconv1x1_s1':
		layers.append(deconv1x1(in_channels, output_channels, stride=1))
	elif type == 'deconv2x2_s2':
		layers.append(deconv2x2(in_channels, output_channels, stride=2))
	elif type == 'deconv3x3_s1':
		layers.append(deconv3x3(in_channels, output_channels, stride=1))
	elif type == 'deconv4x4_s2':
		layers.append(deconv4x4(in_channels, output_channels, stride=2))
	elif type == 'deconv1_s1':
		layers.append(deconv1(in_channels, output_channels, stride=1))
	elif type == 'deconv3_s1':
		layers.append(deconv3(in_channels, output_channels, stride=1))
	elif type == 'deconv4_s2':
		layers.append(deconv4(in_channels, output_channels, stride=2))
	elif type == 'deconv1x1x1_s1':
		layers.append(deconv1x1x1(in_channels, output_channels, stride=1))
	elif type == 'deconv3x3x3_s1':
		layers.append(deconv3x3x3(in_channels, output_channels, stride=1))
	elif type == 'deconv4x4x4_s2':
		layers.append(deconv4x4x4(in_channels, output_channels, stride=2))
	else:
		raise NotImplementedError('layer type [{}] is not implemented'.format(type))

	if  batch_norm  == '1d':
		layers.append(nn.BatchNorm1d(output_channels))
	elif batch_norm == '2d':
		layers.append(nn.BatchNorm2d(output_channels))
	elif batch_norm == '3d':
		layers.append(nn.BatchNorm3d(output_channels))

	if activation == 'relu':
		layers.append(nn.ReLU(inplace=True))
	elif activation == 'sigm':
		layers.append(nn.Sigmoid())
	elif activation == 'leakyrelu':
		layers.append(nn.LeakyReLU(0.2, True))
	else:
		if activation is not None:
			raise NotImplementedError('activation function [{}] is not implemented'.format(activation))

	return nn.Sequential(*layers)

class DoubleConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
	    
        self.conv_layer1    =   _make_layers(in_channels,   mid_channels,   'conv3x3_s1',   '2d')
        self.reluv_layer1   =   nn.ReLU(inplace=True)
        self.conv_layer2    =   _make_layers(mid_channels,  out_channels,   'conv3x3_s1',   '2d')
        self.reluv_layer2   =   nn.ReLU(inplace=True)

    def forward(self, x):
        conv1   = self.conv_layer1(x)
        relu1   = self.reluv_layer1(conv1)
        conv2   = self.conv_layer2(relu1)
        out     = self.reluv_layer2(conv2)
        return out
	
class DoubleConv3D(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
	    
        self.conv_layer1    =   _make_layers(in_channels,   mid_channels,   'conv3x3x3_s1',   '3d')
        self.reluv_layer1   =   nn.ReLU(inplace=True)
        self.conv_layer2    =   _make_layers(mid_channels,  out_channels,   'conv3x3x3_s1',   '3d')
        self.reluv_layer2   =   nn.ReLU(inplace=True)

    def forward(self, x):
        conv1   = self.conv_layer1(x)
        relu1   = self.reluv_layer1(conv1)
        conv2   = self.conv_layer2(relu1)
        out     = self.reluv_layer2(conv2)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, type):
        super().__init__()
        self.type = type
        if type == '2d':
            self.maxpool_conv_2d = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv2D(in_channels, out_channels)
            )
        if type == '3d':
            self.maxpool_conv_3d = nn.Sequential(
                nn.MaxPool3d(2),
                DoubleConv3D(in_channels, out_channels)
            )

    def forward(self, x):
        if self.type=='2d':
            out = self.maxpool_conv_2d(x)
        elif self.type=='3d':
            out = self.maxpool_conv_3d(x)
        else:
            raise NotImplementedError('Down type [{}] is not right'.format(self.type))
        return out


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, type='2d'):
        super().__init__()

        self.type = type

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            if type == '2d':
                self.up_2d = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv_2d = DoubleConv2D(in_channels, out_channels, in_channels // 2)
            if type == '3d':
                self.up_3d = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv_3d = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            if type == '2d':
                self.up_2d = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv_2d = DoubleConv2D(in_channels, out_channels)
            if type == '3d':
                self.up_3d = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv_3d = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        if self.type=='2d':
            x1 = self.up_2d(x1)
			# input is BCHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
							diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            out = self.conv_2d(x)
        elif self.type=='3d':
            x1 = self.up_3d(x1)
			# input is BCXYZ
            diffX = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            diffZ = x2.size()[4] - x1.size()[4]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
							diffY // 2, diffY - diffY // 2,
							diffZ // 2, diffZ - diffZ // 2])
            x = torch.cat([x2, x1], dim=1)
            out = self.conv_3d(x)
        else:
            raise NotImplementedError('Down type [{}] is not right'.format(self.type))
        
        return  out
