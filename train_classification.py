import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import lib.utils as utils

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import time
import lib.layers.base as base_layers
import lib.layers as layers

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument(
    '--data', type=str, default='cifar10', choices=[
        'cifar10',
        'cifar100',
        'mnist',
    ]
)
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./experiments/model-cifar-Resnet18',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=50, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--coeff', type=float, default=0.99)

args = parser.parse_args()

# settings
model_dir = args.model_dir
utils.makedirs(model_dir)
logger = utils.get_logger(logpath=os.path.join(model_dir, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

use_cuda = not args.no_cuda and torch.cuda.is_available()
if use_cuda:
    torch.backends.cudnn.benchmark = True
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}


ACTIVATION_FNS = {
    'identity': base_layers.Identity,
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU,
    'fullsort': base_layers.FullSort,
    'maxmin': base_layers.MaxMin,
    'swish': base_layers.Swish,
    'lcube': base_layers.LipschitzCube,
    'sin': base_layers.Sin,
    'zero': base_layers.Zero,
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, hidden, planes, stride=1):
        super(BasicBlock, self).__init__()

        def build_net():
            layer = nn.Conv2d
            nnet = []
            nnet.append(
                layer(
                    in_planes, hidden, kernel_size=3, stride=1, padding=1, bias=False
                )
            )
            nnet.append(nn.BatchNorm2d(hidden))
            nnet.append(ACTIVATION_FNS['relu']())
            nnet.append(
                layer(
                    hidden, in_planes, kernel_size=3, stride=1, padding=1, bias=False
                )
            )
            nnet.append(nn.BatchNorm2d(in_planes))
            nnet.append(ACTIVATION_FNS['relu']())
            return nn.Sequential(*nnet)

        self.block1 = build_net()
        self.block2 = build_net()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
                ACTIVATION_FNS['relu'](),
            )

    def forward(self, x):
        out = F.relu(x + self.block1(x))
        out = out + self.block2(out)
        out = self.downsample(out)
        return out


class BasicImplicitBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        hidden,
        planes,
        stride=1,
        n_lipschitz_iters=None,
        sn_atol=1e-3,
        sn_rtol=1e-3,
    ):
        super(BasicImplicitBlock, self).__init__()
        coeff = args.coeff
        self.initialized = False

        def build_net():
            layer = base_layers.get_conv2d
            nnet = []
            nnet.append(
                layer(
                    in_planes, hidden, kernel_size=3, stride=1, padding=1, bias=False, coeff=coeff, n_iterations=n_lipschitz_iters, domain=2, codomain=2, atol=sn_atol, rtol=sn_rtol,
                )
            )
            nnet.append(ACTIVATION_FNS['relu']())
            nnet.append(
                layer(
                    hidden, in_planes, kernel_size=3, stride=1, padding=1, bias=False, coeff=coeff, n_iterations=n_lipschitz_iters, domain=2, codomain=2, atol=sn_atol, rtol=sn_rtol,
                )
            )
            nnet.append(ACTIVATION_FNS['relu']())
            return nn.Sequential(*nnet)
        
        self.block = layers.imBlock(
            build_net(),
            build_net(),
        )
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
                ACTIVATION_FNS['relu'](),
            )

    def forward(self, x):
        if self.initialized:
            out = self.block(x)
        else:
            out = self.block(x, restore=True)
            self.initialized = True
        out = self.downsample(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, hidden, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, hidden, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ImplicitResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ImplicitResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, hidden, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, hidden, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

def ImplicitResNet18(num_classes=10):
    return ImplicitResNet(BasicImplicitBlock, [1, 1, 1, 1], num_classes=num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# setup data loader
transform_train = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
transform_test = [
    transforms.ToTensor(),
]

if args.data == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transforms.Compose(transform_train))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transforms.Compose(transform_test))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    args.num_classes = 10
elif args.data == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transforms.Compose(transform_train))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=transforms.Compose(transform_test))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    args.num_classes = 100
elif args.data == 'mnist':
    transform_mnist = [transforms.Pad(2, 0),]
    transform_mnist2 = [lambda x: x.repeat((3, 1, 1))]
    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms.Compose(transform_mnist + transform_train + transform_mnist2))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms.Compose(transform_mnist + transform_test + transform_mnist2))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    args.num_classes = 10

batch_time = utils.RunningAverageMeter(0.97)
loss_meter = utils.RunningAverageMeter(0.97)


def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


def train(args, model, device, train_loader, optimizer, epoch, ema):
    model.train()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target, size_average=False)
        loss_meter.update(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        update_lipschitz(model)
        ema.apply()

        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] | Time {batch_time.val:.3f} | Loss: {loss_meter.val:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), batch_time=batch_time, loss_meter=loss_meter))


def eval_train(model, device, train_loader, ema):
    ema.swap()
    update_lipschitz(model)
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    logger.info('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    ema.swap()
    return train_loss, training_accuracy


def eval_test(model, device, test_loader, ema):
    ema.swap()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    logger.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    ema.swap()
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # model = torch.nn.DataParallel(ResNet18(num_classes=args.num_classes).to(device))
    model = torch.nn.DataParallel(ImplicitResNet18(num_classes=args.num_classes).to(device))
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.to(device)
        model(x)
    ema = utils.ExponentialMovingAverage(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger.info(model)
    logger.info('EMA: {}'.format(ema))
    logger.info(optimizer)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch, ema)

        # evaluation on natural examples
        logger.info('================================================================')
        eval_train(model, device, train_loader, ema)
        eval_test(model, device, test_loader, ema)
        logger.info('================================================================')

        # save checkpoint
        if epoch == args.epochs:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()