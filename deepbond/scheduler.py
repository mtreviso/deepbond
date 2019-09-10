from pathlib import Path

import torch

from deeptagger import constants
from deeptagger import opts
from deeptagger.modules.optim.lr_scheduler import FakeLR


available_schedulers = {
    None: FakeLR,
    'step': torch.optim.lr_scheduler.StepLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'cosine-annealing': torch.optim.lr_scheduler.CosineAnnealingLR
}


def build(options, optim):
    kwargs = {}
    scheduler_class = available_schedulers[options.scheduler]

    if options.scheduler == 'step':
        kwargs['step_size'] = options.step_size
        kwargs['gamma'] = options.gamma

    elif options.scheduler == 'exponential':
        kwargs['gamma'] = options.gamma

    elif options.scheduler == 'cosine-annealing':
        kwargs['t_max'] = options.t_max
        kwargs['eta_min'] = options.eta_min

    return scheduler_class(optim, **kwargs)


def load_state(path, scheduler):
    scheduler_path = Path(path, constants.SCHEDULER)
    scheduler.load_state_dict(torch.load(str(scheduler_path)))


def load(path, optim):
    options = opts.load(path)
    scheduler = build(options, optim)
    load_state(path, scheduler)
    return scheduler


def save(path, optim):
    scheduler_path = Path(path, constants.SCHEDULER)
    torch.save(optim.state_dict(), str(scheduler_path))
