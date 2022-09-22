import torch 
import torch.nn as nn 
import numpy as np 
import tensorly
from tensorly.decomposition import parafac

def cpd_decomposition(conv_layer, rank):

	K = conv_layer.weight.data
	K = K.cpu()
	K = K.numpy()
	#K = K.cuda()
	weight, factor = parafac(K, rank, n_iter_max=100, init='random')

	Kt = factor[0]
	Ks = factor[1]
	Ky = factor[2]
	Kx = factor[3]
	#equation 6
	Us = torch.nn.Conv2d(in_channels=Ks.shape[0], 
                        out_channels=Ks.shape[1], 
                       kernel_size=1, stride=1, padding=0, 
                        dilation=conv_layer.dilation, bias=False)
	#equation 7

	Usy =  torch.nn.Conv2d(in_channels=Ky.shape[1], 
                        out_channels=Ky.shape[1], 
                       kernel_size=(Ky.shape[0], 1),
                        stride=1, padding=(conv_layer.padding[0], 0), 
                       dilation=conv_layer.dilation,
                        groups=Ky.shape[1], bias=False)
	#equation 8
	Usyx = torch.nn.Conv2d(in_channels=Kx.shape[1], 
            out_channels=Kx.shape[1], 
            kernel_size=(1, Kx.shape[0]), stride=conv_layer.stride,
            padding=(0, conv_layer.padding[0]), 
            dilation=conv_layer.dilation, groups=Kx.shape[1], bias=False)
	#equation 9

	Usyxt = torch.nn.Conv2d(in_channels=Kt.shape[1], 
            out_channels=Kt.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=conv_layer.dilation, bias=True)

	#Coppy weight and bias
	Usyxt.bias.data= conv_layer.bias.data

	Us.weight.data = torch.transpose(torch.Tensor(Ks), 1, 0).unsqueeze(-1).unsqueeze(-1)
	Usy.weight.data = torch.transpose(torch.Tensor(Ky), 1, 0).unsqueeze(1).unsqueeze(-1)
	Usyx.weight.data = torch.transpose(torch.Tensor(Kx), 1, 0).unsqueeze(1).unsqueeze(1)
	Usyxt.weight.data = torch.Tensor(Kt).unsqueeze(-1).unsqueeze(-1)

	new_layers = [Us, Usy, Usyx, Usyxt]

	return nn.Sequential(*new_layers)


