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
CHECKPOINT_DIR = "checkpoints/"
os.makedirs(SAVE_DIR, exist_ok=True)

def ensure_dirs_exist():
    """Ensure all necessary directories exist with proper permissions"""
    for directory in [SAVE_DIR, CHECKPOINT_DIR]:
        try:
            os.makedirs(directory, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(directory, "test_write_permission.txt")
            with open(test_file, 'w') as f:
                f.write("Testing write permissions")
            os.remove(test_file)
            print(f"Directory {directory} exists and is writable.")
        except (PermissionError, OSError) as e:
            print(f"ERROR: Cannot create or write to directory {directory}: {e}")
            print(f"Using current directory as fallback")
            return False
    return True

# Check directories at import time
DIRS_OK = ensure_dirs_exist()


class BaseTrainer(pl.Trainer):
    def __init__(self, lightning_module, n_epochs,
                 use_checkpoint_callback=True, checkpoint_path=None,
                 early_stopping_config=None, strategy="null",
                 accelerator='None', profiler='advanced'):

        # Set up directories with better error handling
        self.log_dir = log_dir or os.environ.get("BCCTN_LOG_DIR", "logs/")
        self.checkpoint_dir = checkpoint_dir or os.environ.get("BCCTN_CHECKPOINT_DIR", "checkpoints/")
        # Create directories safely
        for directory in [self.log_dir, self.checkpoint_dir]:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Using directory: {directory}")
            except (PermissionError, OSError) as e:
                # Fall back to project-relative paths
                fallback = os.path.join(os.path.dirname(__file__), os.path.basename(directory))
                print(f"Warning: Cannot use {directory}: {e}")
                print(f"Falling back to: {fallback}")
                os.makedirs(fallback, exist_ok=True)
                
                if directory == self.log_dir:
                    self.log_dir = fallback
                else:
                    self.checkpoint_dir = fallback
                            
        # Check for PyTorch Lightning version compatibility
        import pytorch_lightning as pl
        pl_version = pl.__version__
        print(f"Using PyTorch Lightning version: {pl_version}")
        
        # Check if we have any GPU
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            print(f"Found {gpu_count} GPUs")
        else:
            print("No GPUs found, using CPU")

        if accelerator is None:
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using accelerator: {accelerator}")

        # Only use distributed strategy with multiple GPUs
        strategy_to_use = strategy if gpu_count > 1 else None
        print(f"Using strategy: {strategy_to_use or 'default'}")

        progress_bar = CustomProgressBar(refresh_rate=5)
        early_stopping = EarlyStopping(early_stopping_config["key_to_monitor"],
                                       early_stopping_config["min_delta"],
                                       early_stopping_config["patience_in_epochs"]
                                       )
                                       
        # Create checkpoint directory with explicit checks
        if not os.path.exists(CHECKPOINT_DIR):
            try:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                print(f"Created checkpoint directory: {CHECKPOINT_DIR}")
            except Exception as e:
                print(f"Failed to create checkpoint directory: {e}")
                print("Using current directory instead")
                CHECKPOINT_DIR = "./"
                
        checkpoint_callback = ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            monitor="validation_loss",
            save_last=True,
            save_top_k=3,
            filename="weights-epoch={epoch}-validation_loss={validation_loss:.2f}",
            save_weights_only=True,
            mode="min"
        )
        
        # Verify TensorBoard is installed
        try:
            import tensorboard
            print(f"TensorBoard version: {tensorboard.__version__}")
        except ImportError:
            print("Warning: TensorBoard not found. Install with: pip install tensorboard")
        
        # Create TensorBoard logger with explicit directory
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=SAVE_DIR,
            name="binaural_model",
            log_graph=True,
            default_hp_metric=False
        )
        
        # Also add a CSV logger as backup
        csv_logger = pl_loggers.CSVLogger(
            save_dir=SAVE_DIR,
            name="csv_logs",
            flush_logs_every_n_steps=10  # More frequent flushing
        )

        callbacks = [early_stopping, progress_bar]
        if use_checkpoint_callback:
            callbacks.append(checkpoint_callback)

        super().__init__(
            max_epochs=n_epochs,
            callbacks=callbacks,
            logger=[tb_logger, csv_logger],
            accelerator=accelerator,
            devices=1 if accelerator != "cpu" else None,  # Auto-detect device count
            log_every_n_steps=10,
            enable_progress_bar=True,
            detect_anomaly=True
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
        """Process the collected outputs at the end of an epoch"""
        # Check if outputs is empty
        if not outputs:
            print(f"Warning: No {epoch_type} outputs to process at epoch end.")
            return {}
            
        # Compute epoch metrics
        try:
            outputs = merge_list_of_dicts(outputs)
            
            # Debug: print output keys and shapes
            print(f"{epoch_type} outputs contain keys: {list(outputs.keys())}")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            
            # Calculate basic statistics
            epoch_stats = {}
            if "loss" in outputs:
                epoch_stats[f"{epoch_type}_loss"] = outputs["loss"].mean()
                epoch_stats[f"{epoch_type}_std"] = outputs["loss"].std()
            
            # Log epoch metrics
            for key, value in epoch_stats.items():
                self.log(key, value, on_epoch=True, prog_bar=True)
                print(f"Logged {key}: {value.item() if hasattr(value, 'item') else value}")

            # Save complete epoch data to pickle if requested
            if save_pickle:
                import pickle
                pickle_filename = f"{epoch_type}_epoch_{self.current_epoch}.pickle"
                pickle_path = os.path.join(SAVE_DIR, pickle_filename)
                print(f"Saving epoch data to {pickle_path}")
                try:
                    with open(pickle_path, "wb") as f:
                        pickle.dump(outputs, f)
                    print(f"Successfully saved {pickle_path}")
                except Exception as e:
                    print(f"Error saving pickle file: {e}")

            return epoch_stats
        except Exception as e:
            print(f"Error in {epoch_type} epoch end processing: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def on_train_epoch_end(self):
        """Handle the end of a training epoch"""
        # Store outputs during training steps
        if not hasattr(self, "training_step_outputs") or not self.training_step_outputs:
            print("Warning: No training outputs collected during epoch")
            self.training_step_outputs = []
            return
            
        outputs = self.training_step_outputs
        stats = self._epoch_end(outputs)
        
        # Report training progress
        if stats:
            print(f"Train Epoch {self.current_epoch} End: Loss={stats.get('train_loss', float('nan')):.4f}, Std={stats.get('train_std', float('nan')):.4f}")
        else:
            print(f"Train Epoch {self.current_epoch} End: No stats available")
            
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