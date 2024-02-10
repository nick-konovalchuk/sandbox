from typing import Iterable

import omegaconf
import torch.optim
import torchmetrics
from torch import nn
from tqdm.auto import tqdm


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optimizers: torch.optim.Optimizer,
            device: torch.device,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            metrics: Iterable[torchmetrics.Metric],
            criterion: nn.Module,
            config: omegaconf.DictConfig
    ):
        self.model = model
        self.optimizers = optimizers
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.metrics_train, self.metrics_test = metrics
        self.criterion = criterion
        self.config = config

    def train(self):
        pass

    def _do_train_epoch(self):
        self.model.train()
        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)
            loss = self.criterion(out, y)

            loss.backward()
            for metric in self.metrics_train:
                metric(out, y)
        print("Train: ", end="")
        for metric in self.metrics_train:
            print(metric.compute(), end=", ")
            metric.reset()
        print()

    def _do_test_epoch(self):
        self.model.eval()
        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                out = self.model(x)

            for metric in self.metrics_test:
                metric(out, y)

        print("Train: ", end="")
        for metric in self.metrics_test:
            print(metric.compute(), end="")
            metric.reset()
        print()
