import pytorch_lightning as pl
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import datasets
from torchvision.transforms import v2
import pandas as pd
from pathlib import Path
from pytorch_lightning.loggers import CSVLogger
import augmentations
from torchvision.datasets import CIFAR10
import torchmetrics

pl.seed_everything(42)

train_transform = v2.Compose(
    [
        augmentations.AddParams(),
        augmentations.RandomHorizontalFlip(),
        augmentations.ColorJitter(0.4, 0.4, 0.4, 0.2),
        augmentations.RandomGrayscale(0.5),
        augmentations.RandomRotation(30),
        augmentations.ToTensor(),
    ]
)
test_transform = augmentations.ToTensor()


def get_CIFAR10_dataset():
    dataset = {}
    dataset["train"] = CIFAR10(
        root="./", download=True, train=True, transform=train_transform
    )
    dataset["test"] = CIFAR10(
        root="./", download=True, train=False, transform=test_transform
    )
    return dataset


class MyModel(pl.LightningModule):
    def __init__(self, target: str, guillotine: bool = False, dataset=None):
        """
        target: (str) the name of the dataset target to use to train the backbone
        guillotine: (bool) whether to add a projector and use guillotine (or not)
        """
        super().__init__()
        self.target = target
        self.dataset = dataset
        self.fc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, bias=False, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, bias=False, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        if guillotine:
            self.projector = nn.Sequential(
                nn.Linear(4 * 4 * 128, 4 * 4 * 128, bias=False),
                nn.BatchNorm1d(4 * 4 * 128),
                nn.ReLU(),
                nn.Linear(4 * 4 * 128, 4 * 4 * 128, bias=False),
                nn.BatchNorm1d(4 * 4 * 128),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Identity()
        self.sup_probe = nn.Sequential(nn.Dropout1d(0.2), nn.Linear(4 * 4 * 128, 10))
        self.probe = nn.Linear(4 * 4 * 128, 10)
        self.criterion = CrossEntropyLoss()
        self.evaluate = torchmetrics.classification.MulticlassAccuracy(
            num_classes=10, average=None
        )

    def forward(self, inputs_id):
        outputs = self.fc(inputs_id)
        preds = self.probe(outputs.detach())
        return self.sup_probe(self.projector(outputs)), preds

    def get_losses(self, batch):
        print(batch)
        if self.training:
            input_ids = batch[0][0]
        else:
            input_ids = batch[0]
        labels = batch[1]

        outputs, preds = self(input_ids)
        probe_losses = self.criterion(preds, labels)
        sup_loss = self.criterion(outputs, labels)
        return probe_losses, sup_loss, range(10)

    def training_step(self, batch, batch_idx):
        probe_losses, sup_loss, label_names = self.get_losses(batch)
        loss = probe_losses + sup_loss
        log_dict = {}
        # log_dict = {f"{name}": p for name, p in zip(label_names, probe_losses.tolist())}
        log_dict["target"] = sup_loss.item()
        log_dict["epoch"] = self.current_epoch
        self.loggers[0].log_metrics(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        _, preds = self(batch[0])
        label_names = range(10)
        log_dict = {
            f"{name}": p for name, p in zip(label_names, self.evaluate(preds, batch[1]))
        }
        log_dict["epoch"] = self.current_epoch
        self.loggers[1].log_metrics(log_dict)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), weight_decay=1e-4, lr=1e-4)
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            shuffle=True,
            drop_last=True,
            batch_size=256,
            num_workers=10,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(dataset["test"], batch_size=512, num_workers=10)


if __name__ == "__main__":

    for guillotine in [False, True]:
        target = "rotation"
        dataset = get_CIFAR10_dataset()
        train_results = []
        test_results = []
        model = MyModel(target=target, guillotine=guillotine, dataset=dataset)
        model = torch.compile(model)
        train_logger = CSVLogger(save_dir="lightning_logs", name="train")
        val_logger = CSVLogger(save_dir="lightning_logs", name="val")
        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            logger=[train_logger, val_logger],
            enable_checkpointing=False,
        )
        trainer.fit(model)
        train_metrics = pd.read_csv(Path(train_logger.log_dir) / "metrics.csv")
        print(train_metrics.shape)
        train_metrics = train_metrics.set_index("step").groupby("epoch").mean()
        train_results.append(train_metrics.iloc[[-1]])
        val_metrics = pd.read_csv(Path(val_logger.log_dir) / "metrics.csv")
        print(val_metrics.shape)
        val_metrics = val_metrics.set_index("step").groupby("epoch").mean()
        test_results.append(val_metrics.iloc[[-1]])

        train_results = pd.concat(train_results, axis=0)
        train_results = train_results[TARGETS["dsprite"]]
        train_results.index = TARGETS["dsprite"]

        test_results = pd.concat(test_results, axis=0)
        test_results = test_results[TARGETS["dsprite"]]
        test_results.index = TARGETS["dsprite"]

        train_results.round(4).to_csv(f"./{guillotine}_train.csv")
        test_results.round(4).to_csv(f"./{guillotine}_test.csv")
