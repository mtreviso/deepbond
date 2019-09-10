import math
from torch.optim.lr_scheduler import _LRScheduler


class FakeLR(_LRScheduler):
    """Keep the same learning rate."""

    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        pass

    def get_lr(self):
        pass


class StepOptimizerLRScheduler:
    def __init__(self, **kwargs):
        pass

    def __call__(self, step):
        """
        Calculate the LR scale factor for the current optimization step

        Args:
            step (int): current optimization step

        Returns:
            Factor to scale the learning rate
        """
        raise NotImplementedError


class NoamDecayScheduler(StepOptimizerLRScheduler):
    """Implements learning rate decay from AIAYN paper."""

    def __init__(self, warmup_steps, model_size):
        self.warmup_steps = warmup_steps
        self.model_size = model_size

    def __call__(self, step):
        sqrt_model_size = math.pow(self.model_size, -0.5)
        sqrt_warmup_steps = math.pow(self.warmup_steps, -1.5)
        sqrt_step = math.pow(step, -0.5)
        return sqrt_model_size * min(sqrt_step, step * sqrt_warmup_steps)


class ExpDecayScheduler(StepOptimizerLRScheduler):
    """Adapted from opennmt-py"""

    def __init__(self, initial_lr, decay_steps, start_step=0):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.start_step = start_step

    def __call__(self, step):
        valid_steps = max(0, step - self.start_step + self.decay_steps)
        return math.pow(self.initial_lr, valid_steps // self.decay_steps)


class RsqrtDecayScheduler(StepOptimizerLRScheduler):
    """Adapted from opennmt-py"""

    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return 1.0 / math.sqrt(max(step, self.warmup_steps))
