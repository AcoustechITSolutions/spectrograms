from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self._multiplier = multiplier
        if self._multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self._total_epoch = total_epoch
        self._after_scheduler = after_scheduler
        self._finished = False
        self._last_lr = 0
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self._total_epoch:
            if self._after_scheduler:
                if not self._finished:
                    self._after_scheduler.base_lrs = [base_lr * self._multiplier for base_lr in self.base_lrs]
                    self._finished = True
                    return self._last_lr
                return self._after_scheduler.get_last_lr()
            return [base_lr * self._multiplier for base_lr in self.base_lrs]

        if self._multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self._total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self._multiplier - 1.) * self.last_epoch / self._total_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self._finished and self._after_scheduler:
            if epoch is None:
                self._after_scheduler.step(None)
            else:
                self._after_scheduler.step(epoch - self._total_epoch)
            self._last_lr = self._after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class ConstantWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self._multiplier = multiplier
        self._total_epoch = total_epoch
        self._after_scheduler = after_scheduler
        self._finished = False
        self._last_lr = 0
        super(ConstantWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self._total_epoch:
            if self._after_scheduler:
                if not self._finished:
                    self._finished = True
                return self._after_scheduler.get_last_lr()
        return [base_lr * self._multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self._finished and self._after_scheduler:
            if epoch is None:
                self._after_scheduler.step(None)
            else:
                self._after_scheduler.step(epoch - self._total_epoch)
            self._last_lr = self._after_scheduler.get_last_lr()
        else:
            return super(ConstantWarmupScheduler, self).step(epoch)
