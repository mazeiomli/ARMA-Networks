import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True

from lstms import RowLSTMCell, RowLSTM

batch_size_train = 16
batch_size_test = 16

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
							 transform=torchvision.transforms.Compose([
							   torchvision.transforms.ToTensor(),
							 ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('./files/', train=False, download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
		])),
	batch_size=batch_size_test, shuffle=True)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class MaskedConv1d(nn.Conv1d):
	def __init__(self, *args, mask='B', **kargs):
		super(MaskedConv1d, self).__init__(*args, **kargs)
		assert mask in {'A', 'B'}
		self.mask_type = mask
		self.register_buffer('mask', self.weight.data.clone())
		self.mask.fill_(1)
	
		_, _, W = self.mask.size()
	
		self.mask[:, :, W//2 + (self.mask_type == 'B'):] = 0
	
	def forward(self, x):
		self.weight.data *= self.mask
		return super(MaskedConv1d, self).forward(x)
	
	def forward(self, inputs, initial_state=None):
		n_batch, channel, n_seq, width = inputs.size()
		if initial_state is None:
			hidden_init, cell_init = self.init_state

		else:
			hidden_init, cell_init = initial_state

		states = (hidden_init.repeat(n_batch,1), cell_init.repeat(n_batch, 1))

		steps = []
		for seq in range(n_seq):
			h, states = self.lstm_cell(inputs[:, :, seq, :], states)
			steps.append(h.unsqueeze(1))

		return torch.cat(steps, dim=1).view(-1, n_seq, width, self._hidden_dims).permute(0,3,1,2) # --> (batch, seq_leng

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class ARMA2d(nn.Module):
    def __init__(self, in_channels, out_channels,
            w_kernel_size = 3, w_padding_mode = 'zeros',    
            w_padding = 1, w_stride = 1, w_dilation = 1, w_groups = 1, bias = False,
            a_kernel_size = 3, a_padding_mode = 'circular', 
            a_padding = 0, a_stride = 1, a_dilation = 1):
        """
        Initialization of a 2D-ARMA layer.
        """
        super(ARMA2d, self).__init__()

        self.moving_average = nn.Conv2d(in_channels, out_channels, w_kernel_size,
            padding = w_padding, # padding_mode = w_padding_mode, 
            stride = w_stride, dilation = w_dilation, groups = w_groups, bias = bias) 

        self.autoregressive = AutoRegressive2d(out_channels, a_kernel_size,
                padding = a_padding, padding_mode = a_padding_mode,
            stride = a_stride, dilation = a_dilation)

    def forward(self, x):
        """
        Compuation of the 2D-ARMA layer.
        """
        # size:[M, S, I1, I2]->[M, T, I1, I2]->[M, T, I1, I2]
        x = self.moving_average(x)
        x = self.autoregressive(x)
        return x

class AutoRegressive2d(nn.Module):
    def __init__(self, channels, kernel_size = 3, 
            padding = 0, padding_mode = 'circular',
            stride = 1, dilation = 1):
        """
        Initialization of 2D-AutoRegressive layer.
        """
        super(AutoRegressive2d, self).__init__()

        if padding_mode == "circular":
            self.a = AutoRegressive_circular(
                channels, kernel_size, padding, stride, dilation)

        elif paddind_mode == "reflect":
            self.a = AutoRegressive_reflect(
                channels, kernel_size, padding, stride, dilation)
        else: 
            raise NotImplementedError

    def forward(self, x):
        """
        Computation of 2D-AutoRegressive layer.
        """
        x = self.a(x)
        return x

class AutoRegressive_circular(nn.Module):
    def __init__(self, channels, kernel_size = 3, 
            padding = 0, stride = 1, dilation = 1):
        """
        Initialization of a 2D-AutoRegressive layer (with circular padding).
        """
        super(AutoRegressive_circular, self).__init__()

        self.alpha = nn.Parameter(torch.Tensor(
            channels, kernel_size // 2, 4)) # size:[T, P, 4]

        self.set_parameters()

    def set_parameters(self):
        """
        Initialization of the learnable parameters.
        """
        nn.init.zeros_(self.alpha)

    def forward(self, x):
        """
        Computation of the 2D-AutoRegressive layer (with circular padding). 
        """    
        x = autoregressive_circular(x, self.alpha)
        return x


def autoregressive_circular(x, alpha):
    """
    Computation of a 2D-AutoRegressive layer (with circular padding).
    """

    if  x.size()[-2] < alpha.size()[1] * 2 + 1 or \
        x.size()[-1] < alpha.size()[1] * 2 + 1:
        return x

    # There're 4 chunks, each chunk is [T, P, 1]
    alpha = alpha.tanh() / math.sqrt(2)
    chunks = torch.chunk(alpha, alpha.size()[-1], -1)

    # size: [T, P, 1]
    A_x_left  = (chunks[0] * math.cos(-math.pi / 4) - 
                 chunks[1] * math.sin(-math.pi / 4))

    A_x_right = (chunks[0] * math.sin(-math.pi / 4) +
                 chunks[1] * math.cos(-math.pi / 4))

    A_y_left  = (chunks[2] * math.cos(-math.pi / 4) - 
                 chunks[3] * math.sin(-math.pi / 4))

    A_y_right = (chunks[2] * math.sin(-math.pi / 4) + 
                 chunks[3] * math.cos(-math.pi / 4))

    # zero padding + circulant shift: 
    # [A_x_left 1 A_x_right] -> [1 A_x_right 0 0 ... 0 A_x_left]
    # size: [T, P, 3]->[T, P, I1] or [T, P, I2]
    A_x = torch.cat((torch.ones(chunks[0].size(), device=alpha.device), 
        A_x_right, torch.zeros(chunks[0].size()[0], chunks[0].size()[1],
        x.size()[-2] - 3, device = alpha.device), A_x_left), -1)

    A_y = torch.cat((torch.ones(chunks[2].size(), device = alpha.device), 
        A_y_right, torch.zeros(chunks[2].size()[0], chunks[2].size()[1], 
        x.size()[-1] - 3, device = alpha.device), A_y_left), -1)

    # size: [T, P, I1] + [T, P, I2] -> [T, P, I1, I2]
    A = torch.einsum('tzi,tzj->tzij',(A_x, A_y))

    # Complex Division: FFT/FFT -> irFFT
    A_s = torch.chunk(A, A.size()[1], 1)
    for i in range(A.size()[1]):
        x = ar_circular.apply(x, torch.squeeze(A_s[i], 1))

    return x


def complex_division(x, A, trans_deno=False):
    a, b = torch.chunk(x, 2, -1)
    c, d = torch.chunk(A, 2, -1)

    if trans_deno: 
            # [a bj] / [c -dj] -> [ac-bd/(c^2+d^2) (bc+ad)/(c^2+d^2)j]
        res_l = (a * c - b * d) / (c * c + d * d)
        res_r = (b * c + a * d) / (c * c + d * d)
    else:   # [a bj] / [c  dj] -> [ac+bd/(c^2+d^2) (bc-ad)/(c^2+d^2)j]
        res_l = (a * c + b * d) / (c * c + d * d)
        res_r = (b * c - a * d) / (c * c + d * d)
        
    res = torch.zeros_like(x, device=A.device)

    i = torch.tensor([0], device=A.device)
    res.index_add_(-1, i, torch.unsqueeze(res_l, -1))

    i = torch.tensor([1], device=A.device)
    res.index_add_(-1, i, torch.unsqueeze(res_r,-1))

    return res


def complex_multiplication(x, A, trans_deno=False):
    a, b = torch.chunk(x, 2, -1)
    c, d = torch.chunk(A, 2, -1)

    if trans_deno:
            # [a bj]*[c -dj] -> [ac+bd (bc-ad)j]
        res_l = a * c + b * d
        res_r = b * c - a * d
    else:   # [a bj]*[c  dj] -> [ac-bd (ad+bc)j]
        res_l = a * c - b * d
        res_r = b * c + a * d

    res = torch.zeros_like(x, device=A.device)

    i = torch.tensor([0], device=A.device)
    res.index_add_(-1, i, torch.unsqueeze(res_l, -1))

    i = torch.tensor([1], device=A.device)
    res.index_add_(-1, i, torch.unsqueeze(res_r, -1))

    return res

    
class ar_circular(torch.autograd.Function):

    # x size: [M, T, I1, I2]
    # a size:[T, I1, I2]
    def forward(ctx, x, a):
        X = torch.rfft(x, 2, onesided=False)  # size:[M, T, I1, I2, 2]
        A = torch.rfft(a, 2, onesided=False)  # size:[T, I1, I2, 2]
        Y = complex_division(X, A)  # size:[M, T, I1, I2, 2]
        y = torch.irfft(Y, 2, onesided=False)  # size:[M, T, I1, I2]

        ctx.save_for_backward(A, Y)
        return y

    def backward(ctx, grad_y):
        """
        {grad_a} * a^T    = - grad_y  * y^T
        [T, I1, I2]   * [T, I1, I2] = [M, T, I1, I2] * [M, T, I1, I2]

        a^T    * {grad_x}     = grad_y
        [T, I1, I2] * [M, T, I1, I2]   = [M, T, I1, I2]

        intermediate = grad_y / a^T
        """
        A, Y = ctx.saved_tensors
        grad_x = grad_a = None  

        grad_Y = torch.rfft(grad_y, 2, onesided=False)
        intermediate = complex_division(grad_Y, A, trans_deno=True)  # size:[M,T,I1,I2]
        grad_x = torch.irfft(intermediate, 2, onesided=False)

        intermediate = - complex_multiplication(intermediate, Y, trans_deno=True)# size:[M,T,I1,I2]
        grad_a = torch.irfft(intermediate.sum(0), 2, onesided=False)  #size:[T,I1,I2]
        return grad_x, grad_a

fm = 64
net = nn.Sequential(
    MaskedConv2d('A', 1,  fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    nn.Conv2d(fm, 256, 1))
print net
net.cuda()

tr = data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                     batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
te = data.DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
                     batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
sample = torch.Tensor(144, 1, 28, 28).cuda()
optimizer = optim.Adam(net.parameters())
for epoch in range(25):
    err_tr = []
    cuda.synchronize()
    time_tr = time.time()
    net.train(True)
    for input, _ in tr:
        input = Variable(input.cuda(async=True))
        target = Variable((input.data[:,0] * 255).long())
        loss = F.cross_entropy(net(input), target)
        err_tr.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    cuda.synchronize()
    time_tr = time.time() - time_tr

    err_te = []
    cuda.synchronize()
    time_te = time.time()
    net.train(False)
    for input, _ in te:
        input = Variable(input.cuda(async=True), volatile=True)
        target = Variable((input.data[:,0] * 255).long())
        loss = F.cross_entropy(net(input), target)
        err_te.append(loss.data[0])
    cuda.synchronize()
    time_te = time.time() - time_te

    # sample
    sample.fill_(0)
    net.train(False)
    for i in range(28):
        for j in range(28):
            out = net(Variable(sample, volatile=True))
            probs = F.softmax(out[:, :, i, j]).data
            sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.
    utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=12, padding=0)

    print 'epoch={}; nll_tr={:.7f}; nll_te={:.7f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
        epoch, np.mean(err_tr), np.mean(err_te), time_tr, time_te)
