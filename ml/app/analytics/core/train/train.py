#!/usr/bin/env
import torch
from torch.utils.tensorboard import SummaryWriter
from core.train.eval import evaluate


def train(model, train_loader, eval_loader,
          optimizer, scheduler, criterion, metric,
          model_name, epochs, eval_step_epochs=5,
          save_ckpt_epochs=100, num_classes=1):
    """
    Train loop function.
    :param model: model to train. Subclass of pytorch module
    :param train_loader: instance of pytorch DataLoader class to read data
    :param eval_loader: instance of pytorch DataLoader class to read data
    :param optimizer: pytorch entity optimizer
    :param scheduler: optimizer scheduler to control learning rate
    :param criterion: instance of pytorch loss structures
    :param metric: metric to evaluate. Instance of pytorch metric
    :param model_name: str with model name to save checkpoints
    :param epochs: int with number of epochs to train
    :param eval_step_epochs: make evaluation every 'eval_step_epochs' steps
    :param save_ckpt_epochs: save checkpoint with model state every 'save_ckpt_epochs' epochs
    :param num_classes: number of training classes
    :return trained model. Subclass of pytorch module
    """

    writer = SummaryWriter()

    model.train()
    best_metric_val, metric_val = None, None

    for epoch in range(epochs):
        writer.add_scalar('LR', *scheduler.get_last_lr(), epoch)
        epoch_loss = None
        # train epoch
        for idx, batch in enumerate(train_loader):

            # if batch tuple has 4 elements than this is recurrent type of model
            # otherwise this is simple non sequential model (like CNN or something like this)
            if len(batch) == 3:
                # compute loss
                label, data, pad_lengths = batch
                if num_classes == 1:
                    loss_val = criterion(model.forward((data, pad_lengths)), torch.unsqueeze(label, 1))
                else:
                    loss_val = criterion(model.forward((data, pad_lengths)), label.squeeze().long())
            else:
                label, batch = batch
                target = torch.unsqueeze(label, 1).to(dtype=torch.long)
                loss_val = criterion(model.forward(batch), target)
            # update average epoch loss
            if epoch_loss is None:
                epoch_loss = loss_val
            else:
                n = idx + 1
                epoch_loss = (epoch_loss * n / (n + 1)) + loss_val / (n + 1)
            # compute gradients and make optimization
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        writer.add_scalar('Loss Train', epoch_loss, epoch)
        print(f'Finished epoch number {epoch}')
        # update learning rate
        scheduler.step()
        if epoch != 0 and epoch % eval_step_epochs == 0:
            eval_loss, metric_val = evaluate(model, eval_loader, metric, num_classes, criterion)
            # restore train mode
            model.train()
            writer.add_scalar('Loss Eval', eval_loss, epoch)
            writer.add_scalar('Eval metric', metric_val, epoch)
            print('Model evaluated')
            # save the best checkpoint
            if best_metric_val is None or metric_val >= best_metric_val:
                torch.save(model.state_dict(), f'./runs/best_{model_name}.ckpt')
                best_metric_val = metric_val
        if epoch != 0 and epoch % save_ckpt_epochs == 0:
            torch.save(model.state_dict(), f'./runs/{model_name}_{epoch}.ckpt')
    # save the last checkpoint
    torch.save(model.state_dict(), f'./runs/{model_name}_{epochs}.ckpt')
