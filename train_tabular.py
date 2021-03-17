import argparse
import time
import math
import os
import os.path
import numpy as np
from tqdm import tqdm
import gc

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets

import lib.tabular as tabular
import lib.optimizers as optim
import lib.utils as utils
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts


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

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', type=str, default='gas', choices=[
        'miniboone',
        'gas',
        'hepmass',
        'power',
        'bsds300',
    ]
)
parser.add_argument('--dataroot', type=str, default='data')

parser.add_argument('--coeff', type=float, default=0.9)
parser.add_argument('--vnorms', type=str, default='222222')
parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--epsf', type=float, default=1e-6)

parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='geometric')
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-exact-terms', type=int, default=2)
parser.add_argument('--var-reduc-lr', type=float, default=0)
parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)
parser.add_argument('--brute-force', type=eval, choices=[True, False], default=False)

parser.add_argument('--act', type=str, choices=ACTIVATION_FNS.keys(), default='swish')
parser.add_argument('--dims', type=str, default='128-128-128-128')
parser.add_argument('--nblocks', type=int, default=5)

parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=1000)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)
parser.add_argument('--warmup-iters', type=int, default=0)
parser.add_argument('--annealing-iters', type=int, default=0)
parser.add_argument('--save', help='directory to save results', type=str, default='experiments')
parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=1000)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--begin-epoch', type=int, default=0)

parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=20)
args = parser.parse_args()

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

if device.type == 'cuda':
    logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
else:
    logger.info('WARNING: Using device {}'.format(device))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)


def geometric_logprob(ns, p):
    return torch.log(1 - p + 1e-10) * (ns - 1) + torch.log(p + 1e-10)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x


def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def add_padding(x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


def parallelize(model):
    return torch.nn.DataParallel(model)


logger.info('Loading dataset {}'.format(args.data))
# Dataset and hyperparameters
if args.data == 'miniboone':
    train_dset, _, test_dset = tabular.get_tabular_datasets(args.data, args.dataroot)
    data_dim = 43
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
        drop_last=False,
    )
elif args.data == 'gas':
    train_dset, _, test_dset = tabular.get_tabular_datasets(args.data, args.dataroot)
    data_dim = 8
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
        drop_last=False,
    )
elif args.data == 'hepmass':
    train_dset, _, test_dset = tabular.get_tabular_datasets(args.data, args.dataroot)
    data_dim = 21
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
        drop_last=False,
    )
elif args.data == 'power':
    train_dset, _, test_dset = tabular.get_tabular_datasets(args.data, args.dataroot)
    data_dim = 6
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
        drop_last=False,
    )
elif args.data == 'bsds300':
    train_dset, _, test_dset = tabular.get_tabular_datasets(args.data, args.dataroot)
    data_dim = 63
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
        drop_last=False,
    )

logger.info('Dataset loaded.')
logger.info('Creating model.')

input_size = (args.batchsize, data_dim)
dataset_size = len(train_loader.dataset)

def parse_vnorms():
    ps = []
    for p in args.vnorms:
        if p == 'f':
            ps.append(float('inf'))
        else:
            ps.append(float(p))
    return ps[:-1], ps[1:]

def build_nnet(dims, activation_fn=torch.nn.ReLU):
    nnet = []
    domains, codomains = parse_vnorms()
    for i, (in_dim, out_dim, domain, codomain) in enumerate(zip(dims[:-1], dims[1:], domains, codomains)):
        if i > 0:
            nnet.append(activation_fn())
        nnet.append(
            base_layers.get_linear(
                in_dim,
                out_dim,
                coeff=args.coeff,
                n_iterations=args.n_lipschitz_iters,
                atol=args.sn_tol,
                rtol=args.sn_tol,
                domain=domain,
                codomain=codomain,
                zero_init=(out_dim == data_dim),
            )
        )
    return torch.nn.Sequential(*nnet)


def build_model():
    activation_fn = ACTIVATION_FNS[args.act]
    dims = [data_dim] + list(map(int, args.dims.split('-'))) + [data_dim]
    blocks = []
    for _ in range(args.nblocks):
        blocks.append(
            layers.imBlock(
                build_nnet(dims, activation_fn),
                # ACTIVATION_FNS['zero'](),
                build_nnet(dims, activation_fn),
                n_dist=args.n_dist,
                n_power_series=args.n_power_series,
                exact_trace=False,
                brute_force=args.brute_force,
                n_samples=args.n_samples,
                n_exact_terms=args.n_exact_terms,
                neumann_grad=False,
                grad_in_forward=False, # toy data needn't save memory
                eps_forward=args.epsf
            )
        )
    model = layers.SequentialFlow(blocks).to(device)
    return model

model = build_model()
ema = utils.ExponentialMovingAverage(model)


logger.info(model)
logger.info('EMA: {}'.format(ema))


# Optimization
def tensor_in(t, a):
    for a_ in a:
        if t is a_:
            return True
    return False



if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
elif args.optimizer == 'adamax':
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
else:
    raise ValueError('Unknown optimizer {}'.format(args.optimizer))

best_test_bpd = math.inf
if (args.resume is not None):
    logger.info('Resuming model from {}'.format(args.resume))
    with torch.no_grad():
        x = torch.rand(args.batchsize, data_dim).to(device)
        model(x, restore=True)
    checkpt = torch.load(args.resume)
    sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
    state = model.state_dict()
    state.update(sd)
    model.load_state_dict(state, strict=True)
    ema.set(checkpt['ema'])
    if 'optimizer_state_dict' in checkpt:
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
        # Manually move optimizer state to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    del checkpt
    del state
else:
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.to(device)
        model(x, restore=True)

logger.info(optimizer)

criterion = torch.nn.CrossEntropyLoss()


def compute_loss(x, model, beta=1.0):

    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    z, delta_logp = model(x, zero)

    # compute log p(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz - beta * delta_logp
    loss = -torch.mean(logpx)
    return loss, torch.mean(logpz), torch.mean(-delta_logp)


def estimator_moments(model, baseline=0):
    avg_first_moment = 0.
    avg_second_moment = 0.
    for m in model.modules():
        if isinstance(m, layers.imBlock):
            avg_first_moment += m.last_firmom.item()
            avg_second_moment += m.last_secmom.item()
    return avg_first_moment, avg_second_moment


def compute_p_grads(model):
    scales = 0.
    nlayers = 0
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            scales = scales + m.compute_one_iter()
            nlayers += 1
    scales.mul(1 / nlayers).backward()
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            if m.domain.grad is not None and torch.isnan(m.domain.grad):
                m.domain.grad = None


batch_time = utils.RunningAverageMeter(0.97)
bpd_meter = utils.RunningAverageMeter(0.97)
logpz_meter = utils.RunningAverageMeter(0.97)
deltalogp_meter = utils.RunningAverageMeter(0.97)
firmom_meter = utils.RunningAverageMeter(0.97)
secmom_meter = utils.RunningAverageMeter(0.97)
gnorm_meter = utils.RunningAverageMeter(0.97)
ce_meter = utils.RunningAverageMeter(0.97)


def train(epoch, model):

    model.train()

    total = 0
    correct = 0

    end = time.time()

    for i, (x, y) in enumerate(train_loader):

        global_itr = epoch * len(train_loader) + i
        update_lr(optimizer, global_itr)

        # Training procedure:
        # for each sample x:
        #   compute z = f(x)
        #   maximize log p(x) = log p(z) - log |det df/dx|

        x = x.to(device)

        beta = beta = min(1, global_itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
        bpd, logpz, neg_delta_logp = compute_loss(x, model, beta=beta)

        firmom, secmom = estimator_moments(model)

        bpd_meter.update(bpd.item())
        logpz_meter.update(logpz.item())
        deltalogp_meter.update(neg_delta_logp.item())
        firmom_meter.update(firmom)
        secmom_meter.update(secmom)

        # compute gradient and do SGD step
        loss = bpd
        loss.backward()

        if global_itr % args.update_freq == args.update_freq - 1:

            if args.update_freq > 1:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= args.update_freq

            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            optimizer.zero_grad()
            update_lipschitz(model)
            ema.apply()

            gnorm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            s = (
                'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                'GradNorm {gnorm_meter.avg:.2f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, gnorm_meter=gnorm_meter
                )
            )

            s += (
                ' | Nats {bpd_meter.val:.4f}({bpd_meter.avg:.4f}) | '
                'Logpz {logpz_meter.avg:.0f} | '
                '-DeltaLogp {deltalogp_meter.avg:.0f} | '
                'EstMoment ({firmom_meter.avg:.0f},{secmom_meter.avg:.0f})'.format(
                    bpd_meter=bpd_meter, logpz_meter=logpz_meter, deltalogp_meter=deltalogp_meter,
                    firmom_meter=firmom_meter, secmom_meter=secmom_meter
                )
            )

            logger.info(s)

        del x
        torch.cuda.empty_cache()
        gc.collect()


def validate(epoch, model, ema=None):
    """
    Evaluates the cross entropy between p_data and p_model.
    """
    bpd_meter = utils.AverageMeter()
    ce_meter = utils.AverageMeter()

    if ema is not None:
        ema.swap()

    update_lipschitz(model)

    model.eval()

    correct = 0
    total = 0

    start = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            bpd, _, _ = compute_loss(x, model)
            bpd_meter.update(bpd.item(), x.size(0))

    val_time = time.time() - start

    if ema is not None:
        ema.swap()
    s = 'Epoch: [{0}]\tTime {1:.2f} | Test Nats {bpd_meter.avg:.4f}'.format(epoch, val_time, bpd_meter=bpd_meter)
    logger.info(s)
    return bpd_meter.avg


def get_lipschitz_constants(model):
    lipschitz_constants = []
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            lipschitz_constants.append(m.scale)
        if isinstance(m, base_layers.LopConv2d) or isinstance(m, base_layers.LopLinear):
            lipschitz_constants.append(m.scale)
    return lipschitz_constants


def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


def get_ords(model):
    ords = []
    for m in model.modules():
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            domain, codomain = m.compute_domain_codomain()
            if torch.is_tensor(domain):
                domain = domain.item()
            if torch.is_tensor(codomain):
                codomain = codomain.item()
            ords.append(domain)
            ords.append(codomain)
    return ords


def pretty_repr(a):
    return '[[' + ','.join(list(map(lambda i: f'{i:.2f}', a))) + ']]'


def main(model):
    global best_test_bpd

    last_checkpoints = []
    lipschitz_constants = []
    ords = []

    model = parallelize(model)

    # if args.resume:
    #     validate(args.begin_epoch - 1, model, ema)
    for epoch in range(args.begin_epoch, args.nepochs):

        logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))

        train(epoch, model)
        lipschitz_constants.append(get_lipschitz_constants(model))
        ords.append(get_ords(model))
        logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))
        logger.info('Order: {}'.format(pretty_repr(ords[-1])))

        if args.ema_val:
            test_bpd = validate(epoch, model, ema)
        else:
            test_bpd = validate(epoch, model)

        if test_bpd < best_test_bpd:
            best_test_bpd = test_bpd
            utils.save_checkpoint({
                'state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'ema': ema,
                'test_bpd': test_bpd,
            }, os.path.join(args.save, 'models'), epoch, last_checkpoints, num_checkpoints=5)

        torch.save({
            'state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'ema': ema,
            'test_bpd': test_bpd,
        }, os.path.join(args.save, 'models', 'most_recent.pth'))


if __name__ == '__main__':
    main(model)
