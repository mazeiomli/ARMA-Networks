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

class RowLSTMCell(nn.Module):
	def __init__(self, hidden_dims, image_size, channel_in, *args, **kargs):
		super(RowLSTMCell, self).__init__(*args, **kargs)

		self._hidden_dims = hidden_dims
		self._image_size = image_size
		self._channel_in = channel_in
		self._num_units = self._hidden_dims * self._image_size
		self._output_size = self._num_units
		self._state_size = self._num_units * 2

		self.conv_i_s = MaskedConv1d(self._hidden_dims, 4 * self._hidden_dims, 3, mask='B', padding=_padding(image_size, image_size, 3))
		self.conv_s_s = nn.Conv1d(channel_in, 4 * self._hidden_dims, 3, padding=_padding(image_size, image_size, 3))
   
	def forward(self, inputs, states):
		c_prev, h_prev = states
		h_prev = h_prev.view(-1, self._hidden_dims,  self._image_size)
		inputs = inputs.view(-1, self._channel_in, self._image_size)

		s_s = self.conv_s_s(h_prev) #(batch, 4*hidden_dims, width)
		i_s = self.conv_i_s(inputs) #(batch, 4*hidden_dims, width)



		s_s = s_s.view(-1, 4 * self._num_units) #(batch, 4*hidden_dims*width)
		i_s = i_s.view(-1, 4 * self._num_units) #(batch, 4*hidden_dims*width)

		lstm = s_s + i_s

		lstm = torch.sigmoid(lstm)

		i, g, f, o = torch.split(lstm, (4 * self._num_units)//4, dim=1)

		c = f * c_prev + i * g
		h = o * torch.tanh(c)

		new_state = (c, h)
		return h, new_state

class RowLSTM(nn.Module):
	def __init__(self, hidden_dims, input_size, channel_in, *args, init='zero', **kargs):
		super(RowLSTM, self).__init__(*args, **kargs)
		assert init in {'zero', 'noise', 'variable', 'variable noise'}

		self.init = init
		self._hidden_dims = hidden_dims
		if self.init == 'zero':
			self.init_state = (torch.zeros(1, input_size * hidden_dims), torch.zeros(1, input_size * hidden_dims))
		elif self.init == 'noise':
			self.init_state = (torch.Tensor(1, input_size * hidden_dims), torch.Tensor(1, input_size * hidden_dims))
			nn.init.uniform(self.init_state[0])
			nn.init.uniform(self.init_state[1])  
		elif self.init == 'variable':
			hidden0 = torch.zeros(1,input_size * hidden_dims)
			self._hidden_init_state = torch.nn.Parameter(hidden0, requires_grad=True)
			self._cell_init_state = torch.nn.Parameter(hidden0, requires_grad=True)
			self.init_state = (self._hidden_init_state, self._cell_init_state)
		else:
			hidden0 = torch.Tensor(1, input_size * hidden_dims)
			nn.init.uniform(hidden0)
			self._hidden_init_state = torch.nn.Parameter(hidden0, requires_grad=True)
			self._cell_init_state = torch.nn.Parameter(hidden0, requires_grad=True)
			self.init_state = (self._hidden_init_state, self._cell_init_state)

		self.lstm_cell = RowLSTMCell(hidden_dims, input_size, channel_in)