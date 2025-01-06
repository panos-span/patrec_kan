import torch
import os

from train_utils import train
from evaluation import evaluate
from pathlib import Path


class Training(object):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        save_path,
        device,
        overfit_batch,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.save = save_path
        self.device = device
        self.overfit_batch = overfit_batch
        self.loss_plot_title = f"KViT with {overfit_batch} seeting"

    def train_with_eval(self):
        print(self.loss_plot_title)
        model = train(
            self.model,
            self.train_loader,
            self.val_loader,
            self.optimizer,
            self.epochs,
            self.save,
            self.device,
            self.overfit_batch,
            self.loss_plot_title,
        )
        results = evaluate(model, self.val_loader, self.device, title=f"Confusion Matrix {self.loss_plot_title}")
        return results