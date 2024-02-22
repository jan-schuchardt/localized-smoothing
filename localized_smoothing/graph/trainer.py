import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from localized_smoothing.graph.smooth import SmoothDiscrete


def accuracy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction.cpu() == target.cpu()).float()).item()


class ModelWrapper(pl.LightningModule):

    def __init__(self, model: SmoothDiscrete, trainset, valset, batch_size,
                 learning_rate, learning_rate_decay, weight_decay):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.lr = learning_rate
        self.lr_decay = learning_rate_decay
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        lambda1 = lambda epoch: self.lr_decay**epoch
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
        return [optimizer], [scheduler]

    def execution_step(self, batch):
        attr_idx, edge_idx, targets, indices = batch
        targets = targets.repeat((self.batch_size, 1))
        _, scores = self.model(self.batch_size, attr_idx[0], edge_idx[0])
        loss = self.criterion(scores[:, indices[0], :].permute(0, 2, 1),
                              targets.to(scores.device).long())
        acc = accuracy(scores[:, indices[0], :].argmax(2), targets)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.execution_step(batch)
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, acc = self.execution_step(batch)
            self.log("val_loss", loss)
            self.log("val_acc", acc)
        return loss

    def train_dataloader(self):
        dataloader = DataLoader(
            self.trainset,
            batch_size=1,
            num_workers=4,
            shuffle=True,
            drop_last=False,
        )
        return dataloader

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=1, shuffle=False)
