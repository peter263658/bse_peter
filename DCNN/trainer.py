from omegaconf import OmegaConf
import torch

from torch.optim.lr_scheduler import MultiStepLR
from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
from DCNN.models.model import DCNN
from DCNN.loss import BinauralLoss, Loss
from DCNN.utils.base_trainer import (
    BaseTrainer, BaseLightningModule
)


class DCNNTrainer(BaseTrainer):
    def __init__(self, config):
        lightning_module = DCNNLightningModule(config)
        super().__init__(lightning_module,
                         config["training"]["n_epochs"],
                         early_stopping_config=config["training"]["early_stopping"],
                         checkpoint_path=None,
                        #  strategy=config["training"]["strategy"],
                         accelerator=config["training"]["accelerator"])
                        # accelerator='mps')

    def fit(self, train_dataloaders, val_dataloaders=None):
        super().fit(self._lightning_module, train_dataloaders,
                    val_dataloaders=val_dataloaders)

    def test(self, test_dataloaders):
        super().test(self._lightning_module, test_dataloaders, ckpt_path="best")


class DCNNLightningModule(BaseLightningModule):
    """This class abstracts the
       training/validation/testing procedures
       used for training a DCNN
    """

    def __init__(self, config):
        config = OmegaConf.to_container(config)
        self.config = config

        self.param_mappings = {
            # Original config name → code reference name
            "stoi_weight": ["stoi_weight", "STOI_weight"],
            "snr_loss_weight": ["snr_loss_weight", "snr_weight", "SNR_weight"],
            # Add other parameter variations here
        }
        
        # Create model based on config
        if config["model"]["binaural"]:
            if config["model"]["attention"]:
                model = BinauralAttentionDCNN(**self._get_model_params())
            else:
                model = BinauralAttentionDCNN(**self._get_model_params())
            loss = BinauralLoss(
                ild_weight=self._get_param("ild_weight", default=1),
                ipd_weight=self._get_param("ipd_weight", default=10),
                stoi_weight=self._get_param("stoi_weight", default=10),
                snr_loss_weight=self._get_param("snr_loss_weight", default=1),
                verbose=False,  # Add this to disable console printing
            )
        else:    
            model = DCNN(**self.config["model"])

            loss = Loss(loss_mode=self.config["model"]["loss_mode"],
                        STOI_weight=self.config["model"]["STOI_weight"],
                        SNR_weight=self.config["model"]["snr_weight"])

        super().__init__(model, loss)

    def _get_param(self, name, default=None):
        """Get parameter by name, checking all possible variations"""
        # Check direct match first
        if name in self.config["model"]:
            return self.config["model"][name]
        
        # Check alternative names
        for param, alternatives in self.param_mappings.items():
            if name in alternatives:
                for alt_name in alternatives:
                    if alt_name in self.config["model"]:
                        print(f"Using '{alt_name}' for parameter '{name}'")
                        return self.config["model"][alt_name]
        
        # If not found, warn and return default
        print(f"Warning: Parameter '{name}' not found in config, using default value: {default}")
        return default

    def _get_model_params(self):
        """Extract model parameters from config with validation"""
        # Start with all model parameters
        params = dict(self.config["model"])
        
        # Remove parameters that aren't model initialization args
        non_model_params = ["stoi_weight", "STOI_weight", "snr_weight", "SNR_weight", 
                          "snr_loss_weight", "ild_weight", "ipd_weight"]
        
        for param in non_model_params:
            if param in params:
                del params[param]
        
        return params

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        decay_step = self.config["training"]["learning_rate_decay_steps"]
        decay_value = self.config["training"]["learning_rate_decay_values"]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, decay_step, decay_value)

        return [optimizer], [scheduler]
        
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
        
        # Log individual loss components if available
        if hasattr(self.loss, 'loss_components'):
            for name, value in self.loss.loss_components.items():
                self.log(f"train_{name}_loss", value, 
                        on_step=True, on_epoch=True, prog_bar=False)
        
        # Store output for epoch end processing
        if not hasattr(self, "training_step_outputs"):
            self.training_step_outputs = []
        self.training_step_outputs.append(output)
        
        return output