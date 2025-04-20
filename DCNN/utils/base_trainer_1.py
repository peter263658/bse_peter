import pickle
import os
import pytorch_lightning as pl
import torch

from pytorch_lightning.strategies import DDPStrategy  # Add this import

from pytorch_lightning.callbacks import (TQDMProgressBar,
                                         ModelCheckpoint, EarlyStopping
                                         )
from pytorch_lightning import loggers as pl_loggers

from DCNN.utils.model_utilities import merge_list_of_dicts


# Create log directory if it doesn't exist
SAVE_DIR = "logs/"
os.makedirs(SAVE_DIR, exist_ok=True)


class BaseTrainer(pl.Trainer):
    def __init__(self, lightning_module, n_epochs,
                 use_checkpoint_callback=True, checkpoint_path=None,
                 early_stopping_config=None, strategy="null",
                 accelerator='None', profiler='advanced'):

        gpu_count = torch.cuda.device_count()

        if accelerator is None:
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"

        strategy_to_use = strategy if gpu_count > 1 else None

        progress_bar = CustomProgressBar(refresh_rate=5)
        early_stopping = EarlyStopping(early_stopping_config["key_to_monitor"],
                                       early_stopping_config["min_delta"],
                                       early_stopping_config["patience_in_epochs"]
                                       )
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",  # Specify explicit directory for checkpoints
            monitor="validation_loss",
            save_last=True,
            save_top_k=3,  # Save the top 3 models
            filename="weights-epoch={epoch}-validation_loss={validation_loss:.2f}",
            save_weights_only=True,
            mode="min"  # Since we're monitoring loss
        )
        
        # Create checkpoint directory
        os.makedirs("checkpoints/", exist_ok=True)
        
        # Create TensorBoard logger with explicit log_graph=True
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=SAVE_DIR,
            name="binaural_model",
            log_graph=True,
            default_hp_metric=False  # Don't log hyperparameters by default
        )
        
        # Also add a CSV logger for backup
        csv_logger = pl_loggers.CSVLogger(
            save_dir=SAVE_DIR,
            name="csv_logs"
        )

        callbacks = [early_stopping, progress_bar]
        if use_checkpoint_callback:
            callbacks.append(checkpoint_callback)

        super().__init__(
            max_epochs=n_epochs,
            callbacks=callbacks,
            logger=[tb_logger, csv_logger],  # Use both loggers
            accelerator=accelerator,
            devices=1,  # Explicitly set to 1, not conditionally
            log_every_n_steps=10,  # Log more frequently (default is 50)
            enable_progress_bar=True,
            detect_anomaly=True  # Enable anomaly detection for better debugging
        )

        if checkpoint_path is not None:
            _load_checkpoint(lightning_module.model, checkpoint_path)

        self._lightning_module = lightning_module


class BaseLightningModule(pl.LightningModule):
    """Class which abstracts interactions with Hydra
    and basic training/testing/validation conventions
    """

    def __init__(self, model, loss,
                 log_step=400):
        super().__init__()

        self.is_cuda_available = torch.cuda.is_available()

        self.model = model
        self.loss = loss

        self.log_step = log_step
        
        # Initialize the step counters for better logging
        self.train_step_count = 0
        self.val_step_count = 0
        self.test_step_count = 0

    def _step(self, batch, batch_idx, log_model_output=False,
              log_labels=False):

        x, y = batch
        # 1. Compute model output and loss
        output = self.model(x)
        loss = self.loss(output, y)

        output_dict = {
            "loss": loss
        }

        # Add model output to the output dict if needed
        if log_model_output:
            output_dict["model_output"] = output

        return output_dict

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx, log_model_output=False)
        
        # Increment step counter
        self.train_step_count += 1
        
        # Log loss for each step (in addition to epoch average)
        self.log("train_loss_step", output["loss"], 
                 on_step=True, on_epoch=False, prog_bar=True)
        
        # Log on epoch for averaging
        self.log("train_loss_epoch", output["loss"], 
                 on_step=False, on_epoch=True, prog_bar=False)
        
        # Store output for epoch end processing
        if not hasattr(self, "training_step_outputs"):
            self.training_step_outputs = []
        self.training_step_outputs.append(output)
        
        return output

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx, log_model_output=False, log_labels=True)
        
        # Increment step counter
        self.val_step_count += 1
        
        # Log validation loss
        self.log("validation_loss", output["loss"], 
                 on_step=False, on_epoch=True, prog_bar=True)
        
        # Store output for epoch end processing
        if not hasattr(self, "validation_step_outputs"):
            self.validation_step_outputs = []
        self.validation_step_outputs.append(output)
        
        return output

    def test_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx, log_model_output=False, log_labels=True)
        
        # Increment step counter
        self.test_step_count += 1
        
        # Log test loss
        self.log("test_loss", output["loss"], 
                 on_step=False, on_epoch=True, prog_bar=True)
        
        # Store output for epoch end processing
        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = []
        self.test_step_outputs.append(output)
        
        return output

    def _epoch_end(self, outputs, epoch_type="train", save_pickle=False):
        # Check if outputs is empty
        if not outputs:
            print(f"Warning: No {epoch_type} outputs to process at epoch end.")
            return {}
            
        # Compute epoch metrics
        outputs = merge_list_of_dicts(outputs)
        epoch_stats = {
            f"{epoch_type}_loss": outputs["loss"].mean(),
            f"{epoch_type}_std": outputs["loss"].std()
        }

        # Log epoch metrics
        for key, value in epoch_stats.items():
            self.log(key, value, on_epoch=True, prog_bar=True)

        # Save complete epoch data to pickle if requested
        if save_pickle:
            pickle_filename = f"{epoch_type}.pickle"
            with open(pickle_filename, "wb") as f:
                pickle.dump(outputs, f)

        return epoch_stats

    def on_train_epoch_end(self):
        # Store outputs during training steps
        outputs = self.training_step_outputs if hasattr(self, "training_step_outputs") else []
        stats = self._epoch_end(outputs)
        
        # Report training progress
        if stats:
            print(f"Train Epoch End: Loss={stats['train_loss']:.4f}, Std={stats['train_std']:.4f}")
            
        # Clear the outputs to free memory
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs if hasattr(self, "validation_step_outputs") else []
        stats = self._epoch_end(outputs, epoch_type="validation")
        
        # Report validation progress
        if stats:
            print(f"Validation Epoch End: Loss={stats['validation_loss']:.4f}, Std={stats['validation_std']:.4f}")
            
        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs if hasattr(self, "test_step_outputs") else []
        stats = self._epoch_end(outputs, epoch_type="test", save_pickle=True)
        
        # Report test results
        if stats:
            print(f"Test Results: Loss={stats['test_loss']:.4f}, Std={stats['test_std']:.4f}")
            
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def fit(self, dataset_train, dataset_val):
        super().fit(self.model, dataset_train, val_dataloaders=dataset_val)

    def test(self, dataset_test, ckpt_path="best"):
        super().test(self.model, dataset_test, ckpt_path=ckpt_path)


class CustomProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=1, process_position=0):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        self.enable_progress_bar = True
        self.refresh_rate = refresh_rate
        
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
    
    def disable(self):
        self.enable_progress_bar = False
        
    def enable(self):
        self.enable_progress_bar = True


def _load_checkpoint(model, checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = {}

    for k, v in checkpoint["state_dict"].items():
        k = k.replace("model.", "")
        state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)