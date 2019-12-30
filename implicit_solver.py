from __future__ import print_function

import numpy as np
from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import os
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--num', '-n', default=16, type=int, help='number of input data')
parser.add_argument('--channels', '-c', default=3, type=int, help='number of channels of input data')
parser.add_argument('--height', '-hh', default=32, type=int, help='height of input data')
parser.add_argument('--width', '-w', default=32, type=int, help='width of input data')
parser.add_argument('--kernel', '-k', default=5, type=int, help='kernel size of the Conv layer')
parser.add_argument('--epoch', '-e', default=100, type=int, help='number of epochs for testing')
parser.add_argument('--batch', default=100, type=int, help='batch size for input data')
#parser.addm .._argument('--epoch_num', '-e', default=1, type=int, help='number of epochs while training')
args = parser.parse_args()


torch.set_grad_enabled(False)
class twoModule(nn.Module):
    def __init__(self, KERNEL_SIZE=3, channels=3):
        super(twoModule, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=KERNEL_SIZE, stride=1, padding=int((KERNEL_SIZE-1)/2))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def weight(Conv, Bn):
    w_conv = Conv.weight.clone().view(Conv.out_channels, -1)
    w_bn = torch.diag(Bn.weight.div(torch.sqrt(Bn.eps + Bn.running_var)))
    w = torch.mm(w_bn, w_conv)
    return w

def bias(Conv, Bn):
    if Conv.bias is not None:
        b_conv = Conv.bias
    else:
        b_conv = torch.zeros(Conv.weight.size(0))
    b_bn = Bn.bias - Bn.weight.mul((Bn.running_mean).div(torch.sqrt(Bn.eps + Bn.running_var)))
    b = b_conv + b_bn
    return b


#Test module for randomly generated data
kernel = args.kernel
padding = (kernel - 1) / 2   #ensure that input and output have the same dimensions
height = args.height
width = args.width
height_padding = height + padding * 2
width_padding = width + padding * 2
net = twoModule(args.kernel, args.channels)

for i in range(args.epoch):
    #Usual forward calculation for random generated data
    x = torch.randn(args.num, args.channels, args.height, args.width)
    y = net(x)
    W = weight(net.conv, net.bn)
    B = bias(net.conv, net.bn)


    #Prepare A
    A = torch.zeros(int(args.channels * height * width), int(args.channels * height_padding * width_padding))

    if args.channels == 3:
        w = W.clone().view(3, 3, kernel, kernel)
        for i in range(3):
            for j in range(height):
                for k in range(width):
                    row_index = int(i * height * width + j * width + k)
                    for m in range(3):
                        for n in range(kernel):
                            for p in range(kernel):
                                A[row_index][int(k + n * width_padding + m * width_padding * height_padding + p)] = w[i][m][n][p]
    elif args.channels == 1:
        w = W.clone().view(kernel, -1)
        for j in range(height):
            for k in range(width):
                row_index = int(j * width + k)
                for n in range(kernel):
                    for p in range(kernel):
                        A[row_index][int(k + n * width_padding + p)] = w[n][p]

    Padding = torch.zeros(int(args.channels * height_padding * width_padding), int(args.channels * height * width))
    for m in range(args.channels):
        for i in range(height):
            for j in range(width):
                Padding[int(m * width_padding * height_padding + p * width_padding + i * width_padding + padding + j)][int(m * width * height + i * width + j)] = 1
    AA = torch.mm(A, Padding)


    #Prepare b
    b = y.clone().view(-1)
    for i in range(args.channels):
        for j in range(height):
            for k in range(width):
                b[i * height * width + j * width + k] -= B[i]
    if args.num != 1:
        b = b.clone().view(args.num, -1)


    #Solve Ax=b to solve the implicit problem
    #Prepare the preconditioner
    max_tensor = torch.zeros(int(args.channels * height * width))
    for k in range(int(args.channels * height * width)):
        if abs(torch.max(AA[k]).item()) == 0:
            max_tensor[k] = 0
        else:
            max_tensor[k] = 1.0 / abs(torch.max(AA[k]).item())
    D = torch.diag(max_tensor)

    #Apply the GMRES method
    X = torch.zeros(int(args.num * args.channels * height * width))
    if args.num != 1:
        for i in range(args.num):
            z = gmres(AA.numpy(), b[i].numpy(), tol=1e-06, M=D.numpy())
            for j in range(args.channels * height * width):
                xx = torch.from_numpy(z[0])
                X[i * args.channels * height * width + j] = xx[j]
    else:
        z = gmres(AA.numpy(), b.numpy(), tol=1e-06, M=D.numpy())
        X = torch.from_numpy(z[0])

    XX = X.clone().view(args.num, args.channels, height, width)
    Y = net(XX)
    d = (y - Y).norm(1).item()
    dd = (y - Y).norm(2).item()
    ddd = abs(torch.max(y - Y).item())
    s = (y - Y).norm(1).div(y.norm(1)).item()
    ss = (y - Y).norm(2).div(y.norm(2)).item()
    sss = abs(torch.max(y - Y).item()) / abs(torch.max(y).item())
    print("error_1: %.8f, error_2: %.8f, error_3: %.8f, error_4: %.8f, error_5: %.8f, error_6: %.8f"
          % (d, dd, ddd, s, ss, sss))
    with open(os.path.join('WANTED_PATH' + str(args.num) + '.txt'), 'a') as f:   #you need to modify the code here to get it running
        f.write(str(d))
        f.write(' ')
        f.write(str(dd))
        f.write(' ')
        f.write(str(ddd))
        f.write(' ')
        f.write(str(s))
        f.write(' ')
        f.write(str(ss))
        f.write(' ')
        f.write(str(sss))
        f.write('\n')


'''
#Apply this method to solve the implicit case of CIFAR10
#preparing data from CIFAR10
print("==> preparing data...")
DOWNLOAD = False
if not (os.path.exists('./data/')) or not (os.listdir('./data/')):
    DOWNLOAD = True

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=DOWNLOAD, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=DOWNLOAD, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
kernel = args.kernel
padding = (kernel - 1) / 2   #ensure that input and output have the same dimensions
height = args.height
width = args.width
height_padding = height + padding * 2
width_padding = width + padding * 2


net = twoModule(args.kernel, args.channels)
net.to(device)

batch_idx = 0
for (inputs, targets) in tqdm(testloader):
    batch_idx += 1
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net.forward(inputs)
    W = weight(net.conv, net.bn)
    B = bias(net.conv, net.bn)

    A = torch.zeros(int(args.channels * height * width), int(args.channels * height_padding * width_padding))
    if args.channels == 3:
        w = W.clone().view(3, 3, kernel, kernel)
        for i in range(3):
            for j in range(height):
                for k in range(width):
                    row_index = int(i * height * width + j * width + k)
                    for m in range(3):
                        for n in range(kernel):
                            for p in range(kernel):
                                A[row_index][int(k + n * width_padding + m * width_padding * height_padding + p)] = \
                                w[i][m][n][p]
    elif args.channels == 1:
        w = W.clone().view(kernel, -1)
        for j in range(height):
            for k in range(width):
                row_index = int(j * width + k)
                for n in range(kernel):
                    for p in range(kernel):
                        A[row_index][int(k + n * width_padding + p)] = w[n][p]
    
    Padding = torch.zeros(int(args.channels * height_padding * width_padding), int(args.channels * height * width))
    for m in range(args.channels):
        for i in range(height):
            for j in range(width):
                Padding[int(m * width_padding * height_padding + p * width_padding + i * width_padding + padding + j)][
                    int(m * width * height + i * width + j)] = 1
    AA = torch.mm(A, Padding)

    b = outputs.clone().view(-1)
    for i in range(args.channels):
        for j in range(height):
            for k in range(width):
                b[i * height * width + j * width + k] -= B[i]
    if args.batch != 1:
        b = b.clone().view(args.batch, -1)

    #Solve Ax=b to solve the implicit problem
    #Prepare the preconditioner
    max_tensor = torch.zeros(int(args.channels * height * width))
    for k in range(int(args.channels * height * width)):
        if abs(torch.max(AA[k]).item()) == 0:
            max_tensor[k] = 0
        else:
            max_tensor[k] = 1.0 / abs(torch.max(AA[k]).item())
    D = torch.diag(max_tensor)
    # Apply the GMRES method
    X = torch.zeros(int(args.batch * args.channels * height * width))
    if args.batch != 1:
        for i in range(args.batch):
            z = gmres(AA.cpu().numpy(), b[i].cpu().numpy(), tol=1e-06, M=D.cpu().numpy())
            for j in range(args.channels * height * width):
                xx = torch.from_numpy(z[0])
                X[i * args.channels * height * width + j] = xx[j]
    else:
        z = gmres(AA.cpu().numpy(), b.cpu().numpy(), tol=1e-06, M=D.cpu().numpy())
        X = torch.from_numpy(z[0])
    
    #calculate the numerical error
    XX = X.clone().view(args.batch, args.channels, height, width)
    Y = net(XX)
    d = (outputs - Y).norm(1).item()
    dd = (outputs - Y).norm(2).item()
    ddd = abs(torch.max(outputs - Y).item())
    s = (outputs - Y).norm(1).div(y.norm(1)).item()
    ss = (outputs - Y).norm(2).div(y.norm(2)).item()
    sss = abs(torch.max(outputs - Y).item()) / abs(torch.max(y).item())
    print("error_1: %.8f, error_2: %.8f, error_3: %.8f, error_4: %.8f, error_5: %.8f, error_6: %.8f"
          % (d, dd, ddd, s, ss, sss))
    with open(os.path.join('WANTED_PATH' + str(args.num) + '.txt'), 'a') as f:   #you need to modify the code here to get it running
        f.write(str(d))
        f.write(' ')
        f.write(str(dd))
        f.write(' ')
        f.write(str(ddd))
        f.write(' ')
        f.write(str(s))
        f.write(' ')
        f.write(str(ss))
        f.write(' ')
        f.write(str(sss))
        f.write('\n')
'''