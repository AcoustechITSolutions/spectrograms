#!/usr/bin/env
from torch.optim import SGD, \
                        RMSprop, \
                        Adam
from torch.optim.lr_scheduler import StepLR, \
                                     MultiStepLR, \
                                     ExponentialLR, \
                                     CosineAnnealingLR
from core.train.warmup_scheduler import GradualWarmupScheduler, \
                                        ConstantWarmupScheduler


def create_optimizer(model, config):
    """
    This function is responsible for choosing different optimizer
    algorithms and methods to adjust the learning rate.
    :param model: pytorch model
    :param config: training configuration
    """
    momentum = config.optimizer.momentum if 'optimizer.momentum' in config else 0.9

    types_optimizer = {'SGD': SGD(model.parameters(), lr=config.lr.base_lr),
                       'Momentum': SGD(model.parameters(), lr=config.lr.base_lr, momentum=momentum),
                       'RMSprop': RMSprop(model.parameters(), lr=config.lr.base_lr),
                       'Adam': Adam(model.parameters(), lr=config.lr.base_lr)
                       }
    optimizer = types_optimizer[config.optimizer.type]

    f_epochs = config.num_epochs - config.lr.warmup.epoch\
        if 'lr.warmup.epoch' in config else config.num_epochs
    milestones = config.lr.milestones if 'lr.milestones' in config else [int(f_epochs/4),
                                                                         int(f_epochs/3),
                                                                         int(f_epochs/2)
                                                                         ]
    gamma = config.lr.gamma if 'lr.gamma' in config else 0.95

    types_scheduler = {'cos': CosineAnnealingLR(optimizer, T_max=f_epochs, eta_min=0, last_epoch=-1),
                       'exp': ExponentialLR(optimizer, gamma=gamma),
                       'step': StepLR(optimizer, step_size=f_epochs // 3),
                       'multistep': MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
                       }
    scheduler = types_scheduler[config.lr.scheduler]

    if 'lr.warmup' in config:
        types_warmup = {'constant': ConstantWarmupScheduler,
                        'gradual': GradualWarmupScheduler}
        scheduler = types_warmup[config.lr.warmup.type](
            optimizer,
            config.lr.warmup.multiplier,
            config.lr.warmup.epoch,
            scheduler
        )

    return optimizer, scheduler
