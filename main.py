#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIA Main

@author: Miguel-Ángel Fernández-Torres
"""

import os
import argparse, yaml
from datetime import datetime
from shutil import copyfile

import numpy as np
import torch
import lightning as L
from lightning.pytorch import loggers as loggers
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Lightning module
from lightning_module import DIA

# WandB
import wandb
wandb.login(key='YOUR_WANDB_KEY_HERE')  # Replace with your actual WandB API key


def main(config, config_file, arch, database):
    """Main training/testing/interpretation entry point."""
    
    # Experiment ID and directory setup
    experiment_id = str(datetime.now())
    base_dir = config['trainer']['save_dir']
    
    # Create directory structure
    for subdir in [database, f"{database}/{arch}"]:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
    
    experiment_dir = os.path.join(base_dir, database, arch, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    config['trainer']['save_dir'] = experiment_dir
    
    # Save config files
    copyfile(config_file, os.path.join(experiment_dir, 'config_or.yaml'))
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Build model via Lightning module
    model = DIA(config, arch, database)
    
    # Copy lightning module for reproducibility
    copyfile('./lightning_module.py', os.path.join(experiment_dir, 'lightning_module.py'))
    
    # Load initial weights if specified
    if os.path.isfile(config['arch']['initial_weights']):
        print(f"Loading initial weights: {config['arch']['initial_weights']}")
        checkpoint = torch.load(config['arch']['initial_weights'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('Initial weights loaded!')
    else:
        if config['arch']['initial_weights']:
            print(f"Initial weights not found: {config['arch']['initial_weights']}")
    
    # Save model architecture
    with open(os.path.join(experiment_dir, 'model.txt'), 'w') as f:
        print(model, file=f)
    
    # Save parameter count
    with open(os.path.join(experiment_dir, 'model_parameters.txt'), 'w') as f:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Architecture: {arch}", file=f)
        print(f"Total parameters: {total_params:,}", file=f)
        print(f"Total parameters (M): {total_params/1e6:.2f}M\n", file=f)
        
        for name, param in model.named_parameters():
            print(f"{name}\t{param.numel()}", file=f)
    
    # Interpretation mode
    if config['results']['interpretation']:
        print("Running interpretation mode...")
        model.interpretation(device=config['trainer']['device'])
        return
    
    # Training/testing mode
    # Setup loggers
    wandb_logger = loggers.WandbLogger(
        entity="miguelangelft",
        project='DIA',
        name=f"{arch}_{experiment_id}",
        log_model=True
    )
    logger = [wandb_logger]
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, 'checkpoints'),
        filename='{epoch}-{val_loss:.7f}',
        mode=config['trainer']['mode'],
        monitor=config['trainer']['monitor'],
        save_last=True,
        save_top_k=5,
        save_on_train_epoch_end='train' in config['trainer']['monitor']
    )
    
    early_stopping = EarlyStopping(
        monitor=config['trainer']['monitor_early_stop'],
        min_delta=0,
        patience=config['trainer']['early_stop'],
        verbose=True,
        mode=config['trainer']['mode_early_stop'],
        strict=True,
        check_on_train_epoch_end='train' in config['trainer']['monitor_early_stop']
    )
    
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, early_stopping, ModelSummary(max_depth=-1), lr_logger]
    
    # Resume from checkpoint if specified
    resume = config['arch']['resume'] if os.path.isfile(config['arch']['resume']) else None
    if resume:
        print(f'Resuming from checkpoint: {resume}')
    
    # Create trainer
    trainer = L.Trainer(
        accumulate_grad_batches=1,
        callbacks=callbacks,
        accelerator='gpu' if config['trainer']['num_gpus'] > 0 else 'cpu',
        devices=config['trainer']['num_gpus'],
        limit_train_batches=config['trainer']['limit_train_batches'],
        limit_val_batches=config['trainer']['limit_val_batches'],
        check_val_every_n_epoch=config['trainer']['val_check_interval'],
        logger=logger,
        max_epochs=config['trainer']['epochs'],
        max_steps=config['trainer']['steps'],
        enable_model_summary=True,
        precision=config['trainer']['precision'],
        num_sanity_val_steps=2,
        gradient_clip_val=config['trainer']['gradient_clip_val']
    )
    
    # Training
    print(f"\nTraining {arch} on {database}")
    print(f"Experiment directory: {experiment_dir}")
    trainer.fit(model, ckpt_path=resume)
    
    # Testing
    if config['trainer']['steps'] > 0:
        ckpt_path = 'best'
        trainer.test(ckpt_path=ckpt_path)
    else:
        trainer.test(model, ckpt_path=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DIA Training/Testing/Interpretation")
    parser.add_argument('-c', '--config', default='configs/config.yaml', type=str,
                      help='config file path (default: configs/config.yaml)')
    parser.add_argument('-a', '--arch', default='BaselineMultimodalModel', type=str,
                      help='architecture: BaselineMultimodalModel (default: BaselineMultimodalModel)')
    parser.add_argument('-d', '--database', default='xBDClimate', type=str,
                      help='database name (default: xBDClimate)')
    parser.add_argument('--initial_weights', default="", type=str,
                      help='path to initial weights checkpoint (default: None)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Override initial weights if specified via CLI
    if args.initial_weights:
        config['arch']['initial_weights'] = args.initial_weights
    
    # Run
    main(config, args.config, args.arch, args.database)
