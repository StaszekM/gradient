from datetime import datetime
from typing import List
from src.graph_layering.city_hetero_data import CityHeteroData
from src.graph_layering.data_processing import Normalizer
from torch_geometric.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import pytorch_lightning as pl
from src.lightning.hetero_gnn_module import HeteroGNNModule
from src.lightning.hideable_tqdm_progress_bar import HideableTQDMProgressBar
import torch
from sklearn.metrics import f1_score, roc_auc_score


def train(
    train_data: List[CityHeteroData],
    val_data: List[CityHeteroData],
    test_data: CityHeteroData,
    hparams,
    train_save_dir,
    epochs: int = 100,
    binary: bool =True,
):
    normalizer = Normalizer()

    normalizer.fit(train_data)
    normalizer.transform_inplace([*train_data, *val_data, test_data])

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    model = HeteroGNNModule(
        hetero_data=train_data[0], out_channels=2, add_batch_norm=True, **hparams
    )

    save_dir = train_save_dir + formatted_time

    csv_logger = CSVLogger(save_dir=save_dir, flush_logs_every_n_steps=1)

    progress_bar_callback = HideableTQDMProgressBar(hide_on_valitation=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_last=False,
        dirpath=save_dir + "/checkpoints",
        filename="model-checkpoint-epoch-{epoch:02d}",
        every_n_train_steps=0,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=1,
        callbacks=[progress_bar_callback, checkpoint_callback],
        logger=csv_logger,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        default_root_dir=save_dir,
    )
    trainer.fit(model, train_loader, val_loader)

    model: pl.LightningModule = HeteroGNNModule.load_from_checkpoint(
        checkpoint_callback.best_model_path, hetero_data=train_data[0]
    )
    model.to("cpu")

    with torch.no_grad():
        y_hat = model(
            test_data.x_dict, test_data.edge_index_dict, test_data.edge_attr_dict
        )
        y_hat = torch.softmax(y_hat["hex"], dim=-1)
        if binary:
            f1 = f1_score(
                test_data["hex"].y.cpu().numpy(),
                y_hat.argmax(dim=-1).cpu().numpy(),
                pos_label=1,
                average="binary",
            )

            auc = roc_auc_score(
                test_data["hex"].y.cpu().numpy(),
                y_hat[:, 1].cpu().numpy(),
                average="micro",
            )
        else:
            f1 = f1_score(
                test_data["hex"].y.cpu().numpy(),
                y_hat.argmax(dim=-1).cpu().numpy(),
                pos_label=1,
                average="weighted",
            )

            auc = roc_auc_score(
                test_data["hex"].y.cpu().numpy(),
                y_hat.cpu().numpy(),
                average="micro",
                multi_class='ovr'
            )
        accuracy = (y_hat.argmax(dim=-1) == test_data["hex"].y).sum().item() / len(
            test_data["hex"].y
        )

        csv_logger.log_metrics({"test_auc": float(auc)})
        csv_logger.log_metrics({"test_accuracy": float(accuracy)})
        csv_logger.log_metrics({"test_f1_score": float(f1)})

    csv_logger.finalize("success")

    return auc, accuracy, f1, checkpoint_callback.best_model_path
