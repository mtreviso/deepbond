# this is not exactly a torch module :-/


class StepDecayOptimizer:
    """Simple wrapper that implements learning rate decay during
    optimization steps.

    Args:
        optimizer (nn.optim): torch optimizer object
        lr (float): initial learning rate
        lr_scheduler_fn (optional): a callable that receives the current step
            as argument and returns a factor that will scale the lr
    """
    def __init__(self, optimizer, lr_scheduler_fn=None):
        # Completely wrap optimizer
        self.__class__ = type(optimizer.__class__.__name__,
                              (self.__class__, optimizer.__class__),
                              {})
        self.__dict__ = optimizer.__dict__
        self.optimizer = optimizer
        self._init_lrs = tuple([p['lr'] for p in optimizer.param_groups])
        self._lr_scheduler_fn = lr_scheduler_fn
        self._step = 0

    def step(self):
        self._step += 1
        for p, init_lr in zip(self.optimizer.param_groups, self._init_lrs):
            p['lr'] = self._get_learning_rate(init_lr)
        self.optimizer.step()

    def _get_learning_rate(self, lr):
        if self._lr_scheduler_fn is None:
            return lr
        return lr * self._lr_scheduler_fn(self._step)

    def state_dict(self):
        return {
            'training_step': self._step,
            'init_lrs': self._init_lrs,
            'lr_scheduler_fn': self._lr_scheduler_fn,
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self._step = state_dict['training_step']
        self._init_lrs = state_dict['init_lrs']
        self._lr_scheduler_fn = state_dict['lr_scheduler_fn']
        self.optimizer.load_state_dict(state_dict['optimizer'])


if __name__ == '__main__':

    from .lr_scheduler import (NoamDecayScheduler,
                               RsqrtDecayScheduler,
                               ExpDecayScheduler)
    import numpy as np
    from copy import deepcopy
    from matplotlib import pyplot as plt
    from torch.optim import SGD
    from torch import nn

    # usage:
    # model = MyModel()
    # opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    # noam_scheduler = NoamDecayScheduler(4000, model_hidden_size)
    # step_optim = StepDecayOptimizer(opt, noam_scheduler)

    lr = 1.0
    model = nn.ModuleList([
        nn.Linear(100, 10),
        nn.Linear(10, 10),
        nn.Linear(10, 2)]
    )

    print(model)

    model_params = [
        {'params': model[0].parameters(), 'lr': 1.0},
        {'params': model[1].parameters(), 'nesterov': True},
        {'params': model[2].parameters(), 'lr': 0.001},
    ]
    optim = SGD(model_params, lr=0.3)
    for p in optim.param_groups:
        print(list(filter(lambda x: x[0] != 'params', p.items())))

    opts = [StepDecayOptimizer(deepcopy(optim), NoamDecayScheduler(4000, 512)),
            StepDecayOptimizer(deepcopy(optim), NoamDecayScheduler(8000, 512)),
            StepDecayOptimizer(deepcopy(optim), NoamDecayScheduler(4000, 256)),
            StepDecayOptimizer(deepcopy(optim), RsqrtDecayScheduler(1000)),
            StepDecayOptimizer(deepcopy(optim), ExpDecayScheduler(0.1, 8000))
            ]

    epoch_steps = 20000  # nb of steps for one epoch
    rates = []
    x = np.arange(1, epoch_steps)
    for _ in x:
        rs = []
        for opt in opts:
            opt.step()
            rs.append(opt._get_learning_rate(2.0))
        rates.append(rs)

    # plt.semilogy(x, rates)
    plt.plot(x, rates)
    plt.legend(["noam:512:4000",
                "noam:512:8000",
                "noam:256:4000",
                "rsqrt:1000",
                "exp:0.9:100"])
    plt.show()
