import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from .broyden import broyden
import copy
import lib.layers.base as base_layers

import logging

logger = logging.getLogger()

__all__ = ['imBlock']


def find_fixed_point(g, y, threshold=1000, eps=1e-5):
    x, x_prev = g(y), y
    i = 0
    tol = eps + eps * y.abs()
    while not torch.all((x - x_prev)**2 / tol < 1.):
        x, x_prev = g(x), x
        i += 1
        if i > threshold:
            logger.info(torch.abs(x - x_prev).max())
            logger.info('Iterations exceeded 1000 for fixed point.')
            break
    return x

def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return list(set(primfac))

def choose_prime(n):
    if n == 32 * 32 * 3:
        return 3079
    elif n == 2:
        return 2
    else:
        assert False, 'Please specify the prime or the power of a prime given {}'.format(n)


class RootFind(Function):
    @staticmethod
    def f(nnet_z, nnet_x, z, x):
        return nnet_x(x) - nnet_z(z)

    @staticmethod
    def banach_find_root(nnet_z, nnet_x, z0, x, *args):
        eps = args[-2]
        threshold = args[-1]    # Can also set this to be different, based on training/inference
        x_embed = nnet_x(x) + x
        g = lambda z: x_embed - nnet_z(z)
        z_est = find_fixed_point(g, z0, threshold=threshold, eps=eps)
        if threshold > 100:
            torch.cuda.empty_cache()
        return z_est.clone().detach()

    @staticmethod
    def broyden_find_root(nnet_z, nnet_x, z0, x, *args):
        eps = args[-2]
        threshold = args[-1]    # Can also set this to be different, based on training/inference
        x_embed = nnet_x(x) + x
        g = lambda z: x_embed - nnet_z(z) - z
        result_info = broyden(g, torch.zeros_like(z0).to(z0), threshold=threshold, eps=eps, name="forward")
        if result_info['prot_break']:
            z_est = RootFind.banach_find_root(nnet_z, nnet_x, z0, x, eps, 1000)
        else:
            z_est = result_info['result']
        if threshold > 100:
            torch.cuda.empty_cache()
        return z_est.clone().detach()

    @staticmethod
    def forward(ctx, nnet_z, nnet_x, z0, x, method, *args):
        if method == 'broyden':
            root_find = RootFind.broyden_find_root
        else:
            root_find = RootFind.banach_find_root
        ctx.args_len = len(args)
        with torch.no_grad():
            z_est = root_find(nnet_z, nnet_x, z0, x, *args)

            # If one would like to analyze the convergence process (e.g., failures, stability), should
            # insert here or in broyden_find_root.
            return z_est

    @staticmethod
    def backward(ctx, grad_z):
        assert 0, 'Cannot backward to this function.'
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, None, grad_z, None, *grad_args)


class imBlock(nn.Module):

    def __init__(
        self,
        nnet_x,
        nnet_z,
        geom_p=0.5,
        lamb=2.,
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        n_exact_terms=2,
        n_exact_terms_test=20,
        n_dist='geometric',
        neumann_grad=True,
        grad_in_forward=True,
        eps_forward=1e-6,
        eps_backward=1e-10,
        eps_sample=1e-5,
        threshold=30,
    ):
        """
        Args:
            nnet: a nn.Module
            n_power_series: number of power series. If not None, uses a biased approximation to logdet.
            exact_trace: if False, uses a Hutchinson trace estimator. Otherwise computes the exact full Jacobian.
            brute_force: Computes the exact logdet. Only available for 2D inputs.
        """
        super(imBlock, self).__init__()
        
        self.nnet_x = nnet_x
        self.nnet_z = nnet_z
        self.nnet_x_copy = copy.deepcopy(self.nnet_x)
        self.nnet_z_copy = copy.deepcopy(self.nnet_z)
        for params in self.nnet_x_copy.parameters():
            params.requires_grad_(False)
        for params in self.nnet_z_copy.parameters():
            params.requires_grad_(False)

        self.n_dist = n_dist
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p))).float()
        self.lamb = nn.Parameter(torch.tensor(lamb)).float()
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.n_exact_terms_test = n_exact_terms_test
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad
        self.eps_forward = eps_forward
        self.eps_backward = eps_backward
        self.eps_sample = eps_sample
        self.threshold = threshold

        # store the samples of n.
        self.register_buffer('last_n_samples', torch.zeros(self.n_samples))
        self.register_buffer('last_firmom', torch.zeros(1))
        self.register_buffer('last_secmom', torch.zeros(1))


    class Backward(Function):
        """
        A 'dummy' function that does nothing in the forward pass and perform implicit differentiation
        in the backward pass. Essentially a wrapper that provides backprop for the `imBlock` class.
        You should use this inner class in imBlock's forward() function by calling:
        
            self.Backward.apply(self.func, ...)
            
        """
        @staticmethod
        def forward(ctx, nnet_z, nnet_x, z, x, *args):
            ctx.save_for_backward(z, x)
            ctx.nnet_z = nnet_z
            ctx.nnet_x = nnet_x
            ctx.args = args
            return z

        @staticmethod
        def backward(ctx, grad):
            torch.cuda.empty_cache()

            grad = grad.clone()
            z, x = ctx.saved_tensors
            args = ctx.args
            eps, threshold = args[-2:]

            nnet_z = ctx.nnet_z
            nnet_x = ctx.nnet_x
            z = z.clone().detach().requires_grad_()
            x = x.clone().detach().requires_grad_()

            with torch.enable_grad():
                Fz = nnet_z(z) + z

            def g(x_):
                Fz.backward(x_, retain_graph=True)   # Retain for future calls to g
                xJ = z.grad.clone().detach()
                z.grad.zero_()
                return xJ - grad

            dl_dh = torch.zeros_like(grad).to(grad)
            result_info = broyden(g, dl_dh, threshold=threshold, eps=eps, name="backward")
            dl_dh = result_info['result']
            Fz.backward(torch.zeros_like(dl_dh), retain_graph=False)

            with torch.enable_grad():
                Fx = nnet_x(x) + x
            Fx.backward(dl_dh)
            dl_dx = x.grad.clone().detach()
            x.grad.zero_()

            grad_args = [None for _ in range(len(args))]
            return (None, None, dl_dh, dl_dx, *grad_args)


    def forward(self, x, logpx=None, restore=False):
        z0 = x.clone().detach()
        if restore:
            with torch.no_grad():
                _ = self.nnet_x_copy(z0)
                _ = self.nnet_z_copy(z0)
        z = RootFind.apply(self.nnet_z, self.nnet_x, z0, z0, 'broyden', self.eps_forward, self.threshold)
        z = RootFind.f(self.nnet_z, self.nnet_x, z.detach(), z0) + z0 # For backwarding to parameters in func
        self.nnet_x_copy.load_state_dict(self.nnet_x.state_dict())
        self.nnet_z_copy.load_state_dict(self.nnet_z.state_dict())
        z = self.Backward.apply(self.nnet_z_copy, self.nnet_x_copy, z, x, 'broyden', self.eps_backward, self.threshold)
        if logpx is None:
            return z
        else:
            return z, logpx - self._logdetgrad(z, x)

    def inverse(self, z, logpy=None):
        x0 = z.clone().detach()
        x = RootFind.apply(self.nnet_x, self.nnet_z, x0, z, 'broyden', self.eps_sample, self.threshold)
        # x = RootFind.apply(self.nnet_x, self.nnet_z, x0, z, 'banach', self.eps_sample, self.threshold)
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(z, x)

    def _logdetgrad(self, z, x):
        """Returns logdet|dz/dx|."""

        with torch.enable_grad():
            if (self.brute_force or not self.training) and (x.ndimension() == 2 and x.shape[1] <= 10)
                x = x.requires_grad_(True)
                z = z.requires_grad_(True)
                Fx = x + self.nnet_x(x)
                Jx = batch_jacobian(Fx, x)
                logdet_x = torch.logdet(Jx)

                Fz = z + self.nnet_z(z)
                Jz = batch_jacobian(Fz, z)
                logdet_z = torch.logdet(Jz)

                return (logdet_x - logdet_z).view(-1, 1)
            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)

            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.
            else:
                # Unbiased estimation with more exact terms.

                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + self.n_exact_terms_test
                coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms_test) * \
                    sum(n_samples >= k - self.n_exact_terms_test) / len(n_samples)

            if not self.exact_trace:
                ####################################
                # Power series with trace estimator.
                ####################################
                # vareps_x = torch.randn_like(x)
                # vareps_z = torch.randn_like(z)
                vareps_x = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.5])).sample(x.shape).reshape(x.shape).to(x) * 2 - 1
                vareps_z = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.5])).sample(z.shape).reshape(z.shape).to(z) * 2 - 1

                # Choose the type of estimator.
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

                # Do backprop-in-forward to save memory.
                if self.training and self.grad_in_forward:
                    logdet_x = mem_eff_wrapper(
                        estimator_fn, self.nnet_x, x, n_power_series, vareps_x, coeff_fn, self.training
                    )
                    logdet_z = mem_eff_wrapper(
                        estimator_fn, self.nnet_z, z, n_power_series, vareps_z, coeff_fn, self.training
                    )
                    logdetgrad = logdet_x - logdet_z
                else:
                    x = x.requires_grad_(True)
                    z = z.requires_grad_(True)
                    Fx = self.nnet_x(x)
                    Fz = self.nnet_z(z)
                    logdet_x = estimator_fn(Fx, x, n_power_series, vareps_x, coeff_fn, self.training)
                    logdet_z = estimator_fn(Fz, z, n_power_series, vareps_z, coeff_fn, self.training)
                    logdetgrad = logdet_x - logdet_z
            else:
                ############################################
                # Power series with exact trace computation.
                ############################################
                x = x.requires_grad_(True)
                z = z.requires_grad_(True)
                Fx = self.nnet_x(x)
                Jx = batch_jacobian(Fx, x)
                logdetJx = batch_trace(Jx)
                Jx_k = Jx
                for k in range(2, n_power_series + 1):
                    Jx_k = torch.bmm(Jx, Jx_k)
                    logdetJx = logdetJx + (-1)**(k+1) / k * coeff_fn(k) * batch_trace(Jx_k)
                Fz = self.nnet_z(z)
                Jz = batch_jacobian(Fz, z)
                logdetJz = batch_trace(Jz)
                Jz_k = Jz
                for k in range(2, n_power_series + 1):
                    Jz_k = torch.bmm(Jz, Jz_k)
                    logdetJz = logdetJz + (-1)**(k+1) / k * coeff_fn(k) * batch_trace(Jz_k)
                logdetgrad = logdetJx - logdetJz

            if self.training and self.n_power_series is None:
                self.last_n_samples.copy_(torch.tensor(n_samples).to(self.last_n_samples))
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))
            return logdetgrad.view(-1, 1) 

    def extra_repr(self):
        return 'dist={}, n_samples={}, n_power_series={}, neumann_grad={}, exact_trace={}, brute_force={}, neumann_grad={}, grad_in_forward={}'.format(
            self.n_dist, self.n_samples, self.n_power_series, self.neumann_grad, self.exact_trace, self.brute_force, self.neumann_grad, self.grad_in_forward
        )


def batch_jacobian(g, x, create_graph=True):
    jac = []
    for d in range(g.shape[1]):
        jac.append(torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=create_graph)[0].view(x.shape[0], 1, x.shape[1]))
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)



#####################
# Logdet Estimators
#####################
class MemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        return (None, None, grad_x, None, None, None, None) + grad_params


def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = (-1)**(k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad


def mem_eff_wrapper(estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *list(gnet.parameters())
    )


# -------- Helper distribution functions --------
# These take python ints or floats, not PyTorch tensors.


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)


def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


# -------------- Helper functions --------------


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)


def _flatten(sequence):
    flat = [p.reshape(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [p.reshape(-1) if p is not None else torch.zeros_like(q).view(-1) for p, q in zip(sequence, like_sequence)]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])
