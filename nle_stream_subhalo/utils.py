"""Model construction helpers (optimizer, scheduler, activation, MLP).

Copied from the jgnn package to keep the NLE construction logic aligned
with NPE. Accessors use dict-style `.get`/`[...]` so both plain dicts (e.g.
from a notebook) and `ml_collections.ConfigDict` (from a training config)
work unchanged.
"""

import math
from functools import partial

import torch
import torch.nn as nn


def get_activation(act, act_args=None, return_instance=True):
    """Get an activation callable (class or instance).

    If `return_instance` is False, returns the class (or a partial binding
    `act_args`) — zuko needs a class, while torch modules want an instance.
    """
    key = act.lower().replace('-', '').replace('_', '')
    mapping = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'leakyrelu': nn.LeakyReLU,
        'elu': nn.ELU,
        'prelu': nn.PReLU,
        'selu': nn.SELU,
    }
    act_cls = mapping.get(key)
    if act_cls is None:
        raise ValueError(f'Unknown activation function: {act}')

    if return_instance:
        return act_cls(**act_args) if act_args else act_cls()
    return partial(act_cls, **act_args) if act_args else act_cls


class MLP(nn.Module):
    """MLP with optional batch normalization and dropout.

    Parameters
    ----------
    input_size : int
        Size of the input features
    output_size : int
        Size of the output features
    hidden_sizes : list of int
        Sizes of hidden layers
    act : nn.Module
        Activation instance, e.g. nn.ReLU()
    batch_norm : bool
        Whether to use batch normalization
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes=(512,),
        act=nn.ReLU(),
        batch_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(act)
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    """Cosine annealing learning rate scheduler with warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer
    decay_steps : int
        Initial number of steps for the first cosine cycle (T_0)
    warmup_steps : int
        Number of warmup steps at the beginning
    eta_min : float, default=0
        Minimum learning rate multiplier
    last_epoch : int, default=-1
        The index of last epoch
    restart : bool, default=False
        Whether to restart the cosine annealing after decay_steps
    T_mult : float, default=1
        Multiplicative factor for increasing cycle length after each restart.
    """

    def __init__(self, optimizer, decay_steps, warmup_steps, eta_min=0,
                 last_epoch=-1, restart=False, T_mult=1):
        self.T_0 = decay_steps
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.restart = restart
        self.T_mult = T_mult
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def _get_cycle_length(self, cycle_idx):
        if self.T_mult == 1:
            return self.T_0
        return int(self.T_0 * (self.T_mult ** cycle_idx))

    def _find_cycle(self, step):
        if not self.restart or self.T_mult == 1:
            return step // self.T_0, step % self.T_0, self.T_0

        cumulative_steps = 0
        cycle_idx = 0
        while True:
            cycle_length = self._get_cycle_length(cycle_idx)
            if cumulative_steps + cycle_length > step:
                return cycle_idx, step - cumulative_steps, cycle_length
            cumulative_steps += cycle_length
            cycle_idx += 1

    def lr_lambda(self, step):
        if not self.restart and step >= self.decay_steps:
            step_in_cycle = self.decay_steps
            cycle_length = self.decay_steps
        else:
            _, step_in_cycle, cycle_length = self._find_cycle(step)

        if step_in_cycle < self.warmup_steps:
            return float(step_in_cycle) / float(max(1, self.warmup_steps))

        progress = ((step_in_cycle - self.warmup_steps) /
                    (cycle_length - self.warmup_steps))
        return self.eta_min + 0.5 * (1 + math.cos(math.pi * progress))


def configure_optimizers(parameters, optimizer_args, scheduler_args=None):
    """Configure optimizer and (optional) LR scheduler for Lightning.

    Parameters
    ----------
    parameters : iterable
        Model parameters to optimize.
    optimizer_args : Mapping
        Optimizer config; keys 'name' (Adam/AdamW), 'lr', 'weight_decay'.
    scheduler_args : Mapping, optional
        Scheduler config; 'name' selects the scheduler, remaining keys are
        scheduler-specific. If 'name' is None/absent, no scheduler is used.

    Returns
    -------
    Optimizer, or dict with 'optimizer' and 'lr_scheduler'.
    """
    optimizer_args = optimizer_args or {}
    scheduler_args = scheduler_args or {}

    name = optimizer_args.get('name', 'Adam')
    lr = optimizer_args.get('lr', 1e-3)
    weight_decay = optimizer_args.get('weight_decay', 0.0)

    if name == 'Adam':
        optimizer = torch.optim.Adam(
            parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(
            parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented')

    if scheduler_args.get('name') is None:
        return optimizer

    sched_name = scheduler_args['name']
    if sched_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            factor=scheduler_args['factor'],
            patience=scheduler_args['patience'])
        monitor = scheduler_args.get('monitor', 'val/loss')
    elif sched_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_args['T_max'],
            eta_min=scheduler_args['eta_min'])
        monitor = None
    elif sched_name == 'WarmUpCosineAnnealingLR':
        scheduler = WarmUpCosineAnnealingLR(
            optimizer,
            decay_steps=scheduler_args['decay_steps'],
            warmup_steps=scheduler_args['warmup_steps'],
            eta_min=scheduler_args['eta_min'],
            restart=scheduler_args.get('restart', False),
            T_mult=scheduler_args.get('T_mult', 1))
        monitor = None
    else:
        raise NotImplementedError(f'Scheduler {sched_name} not implemented')

    lr_scheduler_config = {
        'scheduler': scheduler,
        'interval': scheduler_args.get('interval', 'step'),
        'frequency': 1,
    }
    if monitor is not None:
        lr_scheduler_config['monitor'] = monitor

    return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
