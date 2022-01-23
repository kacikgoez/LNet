import torch
from torch.autograd import Function

name = 'BihalfLayer'
num_epochs = 25
batch_size = 128
learning_rate = 1e-3
encode_length = 64    # hash code length
gamma = 1            # parameter γ

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class hash(Function):
    @staticmethod
    def forward(ctx, U):
        # Yunqiang for half and half (optimal transport)
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        B_creat = torch.cat((torch.ones([int(N / 2), D], device=device), -torch.ones([N - int(N / 2), D], device=device))).to(device)
        B = torch.zeros(U.shape, device=device).scatter_(0, index, B_creat)

        ctx.save_for_backward(U, B)

        return B

    @staticmethod
    def backward(ctx, g):
        U, B = ctx.saved_tensors
        add_g = (U - B) / (B.numel())

        grad = g + gamma * add_g

        return grad

def hash_layer(input):
    return hash.apply(input)

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor