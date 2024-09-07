from .group import CyclicGroup
from .utils import *

from layers.liftingconv import LiftingConvolution
from layers.groupconv import GroupConvolution
from models import CNN, GroupEquvariantCNN

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import TrainingArguments

import numpy as np


model_dict = {
    'CNN': CNN,
    'GCNN': GroupEquivariantCNN
}

class DataModule(pl.LightningModule):

    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs
            model_name - Name of the model/CNN to run. Used for creating the model (see functions below)
            model_hparam - Hyperparameters for the model, as dictionary
            optimize_name - Name of the optimizer to use. Currently supported. Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning_rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()


    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        # AdamW is Adam with a correct implementation of weight decay
        optimizer = optim.AdamW(
                                self.parameters(), **self.hparams.optimizer_hparams)

        return [optimizer], []

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim = -1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step = False, on_epoch = True)
        self.log('train_loss', loss)

        return loss # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim = -1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc, prog_bar = True)


    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim = -1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('test_acc', acc, prog_bar = True)



def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \"{model_name}\".Available models are: {str(model_dict.keys())}"

def train_model(model_name, save_name = None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to looj up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.

    """

    if save_name is None:
        save_name = model_name

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name), # Where to save the model
                         accelerator='auto', # We run on a single GPU (if possible)
                         max_epochs = 10,    # How many epochs to train for if no patience is set
                         callbacks = [ModelCheckpoint(save_weights_only = True, mode = 'max', monitor = 'val_acc'), # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                      LearningRateMonitor("epoch")],
                        )

    trainer.logger._default_hp_metric = None # Optinal loggin argument that we don't need

    # Check wether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")

    if os.path.isfile(pretrained_filename):
        print(f"Found model at {pretrained_filename}, loading...")
        model = DataModule.load_from_checkpoint(pretrained_filename) # Automatically loads the model with saved hyperparameters

    else:
        pl.seed_everything(12) # To be reproducable
        model = DataModule(model_name, **kwargs)#.to('mps')
        trainer.fit(model, train_loader, test_loader)
        model = DataModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on test set
    val_result = trainer.test(model.to(device), test_loader, verbose = False)
    result = {'val': val_result[0]['test_acc']}

    return model, result