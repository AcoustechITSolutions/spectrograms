#!/usr/bin/env
import torch


def evaluate(model, eval_loader, metric, num_classes, criterion=None):
    """
    Model evaluation function
    :param model: model to evaluate. Subclass of pytorch module
    :param eval_loader: instance of pytorch DataLoader class to read data
    :param metric: metric to evaluate. Instance of pytorch metric
    :param criterion: instance of pytorch loss structures (optional)
    :return metric value for model calculated on provided data and eval loss if it was specified in arguments
    """
    metric.flush()
    model.eval()
    eval_loss = None
    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):
            # if batch tuple has 4 elements than this is recurrent type of model
            # otherwise this is simple non sequential model (like CNN or something like this)
            if len(batch) == 3:
                label, data, pad_lengths = batch
                preds = model.forward((data, pad_lengths))
            else:
                label, data = batch
                preds = model.forward(data)
            if criterion is not None:
                if eval_loss is None:
                    eval_loss = criterion(preds, torch.unsqueeze(label, 1))
                else:
                    n = idx + 1
                    eval_loss = (eval_loss * n / (n + 1)) + criterion(preds, torch.unsqueeze(label, 1)) / (n + 1)
            # convert probabilities to bin answers using 0.5 threshold
            preds = torch.round(preds)
            metric.update(torch.squeeze(preds), label)

    if eval_loss is None:
        return metric.compute()
    return eval_loss, metric.compute()
