import numpy as np
import torch


class ToTensor(object):
    def __call__(self, data):

        data = torch.from_numpy(data)

        # data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
