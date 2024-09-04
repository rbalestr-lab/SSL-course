import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import datasets
from torchvision.transforms import v2
import pandas as pd
from pathlib import Path
from pytorch_lightning.loggers import CSVLogger

pl.seed_everything(42)

TARGETS = {
    "dsprite": [
        "value_orientation",
        "value_scale",
        "value_shape",
        "value_x_position",
        "value_y_position",
    ]
}

train_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
test_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def train_transforms(examples):
    examples["image"] = [
        train_transform(img.convert("RGB")) for img in examples["image"]
    ]
    return examples


def test_transforms(examples):
    examples["image"] = [
        test_transform(img.convert("RGB")) for img in examples["image"]
    ]
    return examples


def get_dsprites_dataset():
    dataset = datasets.load_dataset("eurecom-ds/dsprites").remove_columns(
        [
            "label_orientation",
            "label_scale",
            "label_shape",
            "label_x_position",
            "label_y_position",
        ]
    )
    dataset = dataset["train"].train_test_split(test_size=0.5)
    for target in TARGETS["dsprite"]:
        mean = dataset["train"].with_format("pandas")[target].mean()
        std = dataset["train"].with_format("pandas")[target].std()
        dataset["train"] = dataset["train"].map(lambda row:{target: (row[target] - mean)/(1e-5+std)}, batched=True, batch_size=2048)
        dataset["test"] = dataset["test"].map(lambda row:{target: (row[target] - mean)/(1e-5+std)}, batched=True, batch_size=2048)
    dataset["train"] = dataset["train"].with_transform(train_transforms)
    dataset["test"] = dataset["test"].with_transform(test_transforms)
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
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
        )
        if guillotine:
            self.projector = nn.Sequential(
                nn.Linear(4*4*128, 4*4*128, bias=False),
                nn.BatchNorm1d(4*4*128),
                nn.ReLU(),
                nn.Linear(4*4*128, 4*4*128, bias=False),
                nn.BatchNorm1d(4*4*128),
                nn.ReLU(),
            )
        else:
            self.projector = nn.Identity()
        self.sup_probe = nn.Sequential(nn.Dropout1d(0.2), nn.Linear(4*4*128, 1))
        self.probe = nn.Linear(4*4*128, 5)
        self.criterion = MSELoss(reduction="none")

    def forward(self, inputs_id):
        outputs = self.fc(inputs_id)
        preds = self.probe(outputs.detach())
        return self.sup_probe(self.projector(outputs)), preds



    def get_losses(self, batch):
        input_ids = batch["image"]
        label_names = [name for name in batch.keys() if name != "image"]
        labels = torch.stack([batch[name] for name in label_names], 1)

        outputs, preds = self(input_ids)
        probe_losses = self.criterion(preds, labels.float()).mean(0)
        sup_loss = self.criterion(
            outputs.squeeze(), batch[self.target].squeeze().float()
        ).mean()
        return probe_losses, sup_loss, label_names

    def training_step(self, batch, batch_idx):
        probe_losses, sup_loss, label_names = self.get_losses(batch)
        loss = probe_losses.mean() + sup_loss
        log_dict = {f"{name}": p for name, p in zip(label_names, probe_losses.tolist())}
        log_dict["target"] = sup_loss.item()
        log_dict["epoch"] = self.current_epoch
        self.loggers[0].log_metrics(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        probe_losses, _, label_names = self.get_losses(batch)
        log_dict = {f"{name}": p for name, p in zip(label_names, probe_losses.tolist())}
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
    dataset = get_dsprites_dataset()

    for guillotine in [False, True]:
        train_results = []
        test_results = []
        for target in TARGETS["dsprite"]:
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
