"""
Lightning module wrapper for DIA models.

Handles training, validation, testing, and interpretation across all model architectures.
"""

import os
# Fix OpenCV Qt plugin issue in headless environments
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tqdm

# Captum for integrated gradients
from captum.attr import IntegratedGradients

# Import model registry
from models import create_model


def make_fullscreen_figure(nrows=1, ncols=1, dpi=100, constrained=True, fallback_size=(16, 9)):
    """Helper to create fullscreen matplotlib figures."""
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        figsize = (screen_width / dpi, screen_height / dpi)
    except Exception:
        # Fallback for headless environments or when tkinter unavailable
        figsize = fallback_size
    return plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, 
                        constrained_layout=constrained)


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def calculate_f1_score(outputs, labels):
    """Calculate F1 macro and weighted scores."""
    from sklearn.metrics import f1_score
    preds = outputs.argmax(dim=1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    return f1_macro, f1_weighted


class CaptumWrapper(nn.Module):
    """Wrapper for Captum to extract specific output from model."""
    def __init__(self, model, output_idx=0):
        super().__init__()
        self.model = model
        self.output_idx = output_idx

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        if isinstance(outputs, (tuple, list)):
            return outputs[self.output_idx]
        return outputs


class DIA(L.LightningModule):
    """
    Lightning module for Disaster Impact Assessment.
    
    Supports multiple architectures via model registry:
    - DamageClimateModel (original, slow)
    - EfficientDamageClimateModel (fast, no attention)
    - LightweightAttentionModel (fast + interpretable, recommended)
    """
    
    def __init__(self, cfg, arch, database):
        super().__init__()
        self.cfg = cfg
        self.arch = arch
        self.database = database
        
        # Create model via registry
        self.model = create_model(
            arch_name=arch,
            dropout_rate=cfg['arch']['dropout_rate'],
            backbone=cfg['arch']['backbone'],
            num_classes=cfg['arch']['num_classes'],
            vis_dim=cfg['arch'].get('vis_dim', 256),
            climate_dim=cfg['arch'].get('climate_dim', 128),
        )
        
        # Load dataset
        if database == 'xBDClimate':
            from databases.xBDClimate_database import get_dataloaders
            self.data_train, self.data_val, self.data_test = get_dataloaders(
                cache_dir=cfg['data']['cache_dir'],
                batch_size=cfg['data']['batch_size'],
                num_workers=cfg['data']['num_workers'],
                augment=cfg['data']['augment'],
            )
        else:
            raise ValueError(f"Unknown database: {database}")
        
        # Class weights for focal loss
        self.class_weights = torch.tensor(cfg['arch']['class_weights'], dtype=torch.float32)
        
        # Storage for batch outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, patches_pre, patches_post, mask_patches, climate_series, 
                event_labels, labels_pre, labels_post, train=False, score=True):
        """
        Forward pass with loss computation.
        
        Returns dict with: loss, f1_macro, f1_weighted, outputs, labels_post, att_weights
        """
        criterion = FocalLoss(alpha=self.class_weights.to(self.device), gamma=2.0)
        loss = torch.tensor([0.0], requires_grad=train).to(self.device)
        
        # Model forward - handle both single and tuple returns
        model_out = self.model(patches_pre, patches_post, climate_series, event_labels)
        if isinstance(model_out, tuple):
            outputs, att_weights = model_out
        else:
            outputs = model_out
            att_weights = None
        
        loss = criterion(outputs, labels_post)
        
        # Evaluation metrics
        if score:
            f1_macro, f1_weighted = calculate_f1_score(outputs, labels_post)
        else:
            f1_macro, f1_weighted = np.nan, np.nan
        
        return {
            'loss': loss,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'outputs': outputs,
            'labels_post': labels_post,
            'att_weights': att_weights,
        }

    def log_metrics(self, outputs, mode='train'):
        """Log metrics to WandB and Lightning."""
        global_step = self.current_epoch
        loss_metrics = ['loss', 'f1_macro', 'f1_weighted']
        
        for metric_name in loss_metrics:
            metric = torch.Tensor([batch[metric_name] for batch in outputs]).nanmean()
            from lightning.pytorch.loggers import WandbLogger
            if self.logger is not None and isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({f"{mode}/{metric_name}": metric, "epoch": global_step})
            self.log(f"{mode}_{metric_name}", metric, on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        patches_pre, patches_post, mask_patches, climate_series, event_labels, labels_pre, labels_post, _, _ = batch
        out = self(patches_pre, patches_post, mask_patches, climate_series, 
                  event_labels, labels_pre, labels_post, train=True)
        self.training_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        self.log_metrics(self.training_step_outputs, mode='train')
        
        # Confusion matrix
        all_outputs = torch.cat([out["outputs"].detach().cpu() for out in self.training_step_outputs], dim=0)
        all_labels = torch.cat([out["labels_post"].detach().cpu() for out in self.training_step_outputs], dim=0)
        preds = all_outputs.argmax(dim=1).numpy()
        targets = all_labels.numpy()
        
        cm = confusion_matrix(targets, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=list(self.data_train.dataset.damage_classes.keys()),
                    yticklabels=list(self.data_train.dataset.damage_classes.keys()),
                    ax=ax, cbar_kws={'label': 'Normalized Count'})
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('Confusion Matrix (Train)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Log to W&B as image (CSVLogger has no image logging)
        from lightning.pytorch.loggers import WandbLogger
        if self.logger is not None and isinstance(self.logger, WandbLogger):
            try:
                import wandb
                self.logger.experiment.log({
                    "train/confusion_matrix": wandb.Image(fig)
                })
            except Exception as e:
                print(f"Warning: Could not log confusion matrix to W&B: {e}")
        plt.close(fig)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        patches_pre, patches_post, mask_patches, climate_series, event_labels, labels_pre, labels_post, _, _ = batch
        out = self(patches_pre, patches_post, mask_patches, climate_series,
                  event_labels, labels_pre, labels_post, train=False)
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        self.log_metrics(self.validation_step_outputs, mode='val')
        
        all_outputs = torch.cat([out["outputs"].detach().cpu() for out in self.validation_step_outputs], dim=0)
        all_labels = torch.cat([out["labels_post"].detach().cpu() for out in self.validation_step_outputs], dim=0)
        preds = all_outputs.argmax(dim=1).numpy()
        targets = all_labels.numpy()
        
        cm = confusion_matrix(targets, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=list(self.data_train.dataset.damage_classes.keys()),
                    yticklabels=list(self.data_train.dataset.damage_classes.keys()),
                    ax=ax, cbar_kws={'label': 'Normalized Count'})
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('Confusion Matrix (Val)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Log to W&B as image (CSVLogger has no image logging)
        from lightning.pytorch.loggers import WandbLogger
        if self.logger is not None and isinstance(self.logger, WandbLogger):
            try:
                import wandb
                self.logger.experiment.log({
                    "val/confusion_matrix": wandb.Image(fig)
                })
            except Exception as e:
                print(f"Warning: Could not log confusion matrix to W&B: {e}")
        
        plt.close(fig)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        patches_pre, patches_post, mask_patches, climate_series, event_labels, labels_pre, labels_post, _, _ = batch
        out = self(patches_pre, patches_post, mask_patches, climate_series,
                  event_labels, labels_pre, labels_post, train=False)
        self.test_step_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        self.log_metrics(self.test_step_outputs, mode='test')
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg['optimizer']['lr'],
            weight_decay=self.cfg['optimizer']['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg['trainer']['epochs'],
            eta_min=self.cfg['optimizer']['lr'] * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

    def train_dataloader(self):
        return self.data_train

    def val_dataloader(self):
        return self.data_val

    def test_dataloader(self):
        return self.data_test

    def interpretation(self, device='cuda'):
        """
        Run interpretation with Integrated Gradients.
        
        Generates satellite + climate attribution plots for target damage class.
        """
        # Disable augmentation
        target_cat = 0  # 0,1,2,3
        from databases.xBDClimate_database import FilteredSubset
        self.data_train = FilteredSubset(self.data_train.dataset, target_cat=target_cat)
        
        train_len = len(self.data_train)
        bs = 128
        train_loader = DataLoader(self.data_train, batch_size=bs, shuffle=True)
        
        # Integrated Gradients
        wrapped_model = CaptumWrapper(self.model, output_idx=0)
        ig = IntegratedGradients(wrapped_model)
        
        for b in tqdm.tqdm(range(len(train_loader))):
            batch = next(iter(train_loader))
            (patches_pre_all, patches_post_all, mask_patches_all, climate_series_all,
             event_labels_all, labels_pre_all, labels_post_all, event_name_all, patches_idx) = batch
            
            # Get model predictions and attention weights
            out = self(patches_pre_all, patches_post_all, mask_patches_all,
                      climate_series_all, event_labels_all, labels_pre_all,
                      labels_post_all, train=False, score=False)
            
            for s in range(bs):
                if s >= len(patches_idx):
                    break
                    
                pred_class = int(torch.argmax(F.softmax(out['outputs'][s])).detach().cpu().numpy())
                if pred_class != target_cat:
                    continue
                
                print(event_name_all[s])
                print(f"Sample {int(patches_idx[s])}")
                print(F.softmax(out['outputs'][s]))
                
                # Compute attributions
                baseline_patches_pre = torch.zeros_like(patches_pre_all[[s]])
                baseline_patches_post = torch.zeros_like(patches_post_all[[s]])
                baseline_climate_series = torch.zeros_like(climate_series_all[[s]])
                
                attr_aux = ig.attribute(
                    inputs=(patches_pre_all[[s]], patches_post_all[[s]], climate_series_all[[s]]),
                    baselines=(baseline_patches_pre, baseline_patches_post, baseline_climate_series),
                    additional_forward_args=(event_labels_all[[s]]),
                    target=pred_class,
                    n_steps=50,
                    internal_batch_size=1
                )
                attr_pre, attr_post, attr_climate = attr_aux
                
                # Handle spatial dimensions in climate if present
                if len(attr_climate.shape) > 3:
                    climate_series = torch.nanmean(climate_series_all[[s]], dim=(3,4)).permute(0,2,1)
                    attr_climate = torch.nanmean(attr_climate, dim=(3,4)).permute(0,2,1)
                else:
                    climate_series = climate_series_all[[s]]
                
                # ============================================================
                # Plot 1: Satellite images with RGB attributions
                # ============================================================
                fig, ax = make_fullscreen_figure(nrows=2, ncols=4, dpi=100, constrained=True, fallback_size=(16, 9))
                
                # Row 1 (Pre)
                ax[0,0].imshow(((patches_pre_all[s]*3*self.data_train.patch_std.reshape(-1,1,1)+
                                self.data_train.patch_mean.reshape(-1,1,1)).permute(1,2,0).detach().cpu().numpy().astype('uint8')))
                ax[0,0].set_title('Pre-image')
                
                im = ax[0,1].imshow(attr_pre[0][0].detach().cpu().numpy())
                fig.colorbar(im, ax=ax[0,1])
                ax[0,1].set_title('Attr Pre R')
                
                im = ax[0,2].imshow(attr_pre[0][1].detach().cpu().numpy())
                fig.colorbar(im, ax=ax[0,2])
                ax[0,2].set_title('Attr Pre G')
                
                im = ax[0,3].imshow(attr_pre[0][2].detach().cpu().numpy())
                fig.colorbar(im, ax=ax[0,3])
                ax[0,3].set_title('Attr Pre B')
                
                # Row 2 (Post)
                ax[1,0].imshow(((patches_post_all[s]*3*self.data_train.patch_std.reshape(-1,1,1)+
                                self.data_train.patch_mean.reshape(-1,1,1)).permute(1,2,0).detach().cpu().numpy().astype('uint8')))
                ax[1,0].set_title('Post-image')
                
                im = ax[1,1].imshow(attr_post[0][0].detach().cpu().numpy())
                fig.colorbar(im, ax=ax[1,1])
                ax[1,1].set_title('Attr Post R')
                
                im = ax[1,2].imshow(attr_post[0][1].detach().cpu().numpy())
                fig.colorbar(im, ax=ax[1,2])
                ax[1,2].set_title('Attr Post G')
                
                im = ax[1,3].imshow(attr_post[0][2].detach().cpu().numpy())
                fig.colorbar(im, ax=ax[1,3])
                ax[1,3].set_title('Attr Post B')
                
                plt.suptitle(f"{event_name_all[s].split('/')[-1][:-3]} - "
                           f"GT: {list(self.data_train.damage_classes.values())[labels_post_all[s]]}, "
                           f"{list(self.data_train.damage_classes.keys())[labels_post_all[s]]} - "
                           f"pred={F.softmax(out['outputs'][s]).detach().cpu().numpy()}")
                
                fig.savefig(f"{self.cfg['trainer']['save_dir']}/{event_name_all[s].split('/')[-1][:-3]}_"
                          f"p{int(patches_idx[s])}_satellite.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                
                # ============================================================
                # Plot 2: Climate attributions and attention weights
                # ============================================================
                event_labels = event_labels_all[[s]].int().cpu().numpy()
                fig, ax = make_fullscreen_figure(nrows=3, ncols=1, dpi=100, constrained=True, fallback_size=(16, 9))
                
                # Climate attributions
                ax[0].plot(attr_climate[0])
                legend1 = ax[0].legend(self.data_train.climate_variable_names, 
                                      title="Attributions", loc="upper right")
                
                # Climate input features
                ax[1].plot(climate_series[0])
                legend2 = ax[1].legend(self.data_train.climate_variable_names,
                                      title="Input features", loc="upper right")
                
                # Attention weights (if available)
                if out['att_weights'] is not None:
                    ax[2].plot(out['att_weights'][s].detach().cpu().numpy())
                    legend3 = ax[2].legend(['Attention weights'], loc="upper right")
                else:
                    ax[2].text(0.5, 0.5, 'No attention weights available',
                              ha='center', va='center', transform=ax[2].transAxes)
                    legend3 = None
                
                # Shade background by event labels
                event_category = ['no event', 'event', 'pre-image', 'post_image', 'climate_end']
                colors = ["orange", "blue", "green", "pink", "red"]
                x = np.arange(event_labels.shape[1])
                start = 0
                event_flag = 0
                lines, lines2, lines3 = [], [], []
                
                # First segment
                if event_labels[0,start] > 0:
                    for a_idx in [0, 1, 2]:
                        l = ax[a_idx].axvline(x[start], color=colors[event_labels[0,start]],
                                             linestyle="--", alpha=0.7, 
                                             label=event_category[event_labels[0,start]])
                        if a_idx == 0: lines.append(l)
                        elif a_idx == 1: lines2.append(l)
                        else: lines3.append(l)
                
                for i in range(1, event_labels.shape[1]):
                    if event_labels[0,i] != event_labels[0,start] and event_labels[0,i] > 0:
                        if event_labels[0,start] == 1 and not event_flag:
                            end = np.where(event_labels==1)[1][-1]
                            for a_idx in [0, 1, 2]:
                                ax[a_idx].axvspan(x[start], x[end], 
                                                color=colors[event_labels[0,start]], alpha=0.3)
                                for pos in [start, end, i]:
                                    l = ax[a_idx].axvline(x[pos], 
                                                        color=colors[event_labels[0,pos]],
                                                        linestyle="--", alpha=0.7,
                                                        label=event_category[event_labels[0,pos]])
                                    if a_idx == 0: lines.append(l)
                                    elif a_idx == 1: lines2.append(l)
                                    else: lines3.append(l)
                            event_flag = 1
                        elif event_labels[0,i] != 1:
                            for a_idx in [0, 1, 2]:
                                l = ax[a_idx].axvline(x[i], color=colors[event_labels[0,i]],
                                                    linestyle="--", alpha=0.7,
                                                    label=event_category[event_labels[0,i]])
                                if a_idx == 0: lines.append(l)
                                elif a_idx == 1: lines2.append(l)
                                else: lines3.append(l)
                        start = i
                
                # Last segment
                if event_labels[0,-1] > 0:
                    for a_idx in [0, 1, 2]:
                        l = ax[a_idx].axvline(x[-1], color=colors[event_labels[0,-1]],
                                            linestyle="--", alpha=0.7,
                                            label=event_category[event_labels[0,-1]])
                        if a_idx == 0: lines.append(l)
                        elif a_idx == 1: lines2.append(l)
                        else: lines3.append(l)
                
                # Add legends
                if legend3 is not None:
                    ax[2].add_artist(legend3)
                ax[2].legend(handles=lines3, title="Events", loc="lower right")
                
                plt.suptitle(f"{event_name_all[s].split('/')[-1][:-3]} - "
                           f"GT: {list(self.data_train.damage_classes.values())[labels_post_all[s]]}, "
                           f"{list(self.data_train.damage_classes.keys())[labels_post_all[s]]} - "
                           f"pred={F.softmax(out['outputs'][s]).detach().cpu().numpy()}")
                
                fig.savefig(f"{self.cfg['trainer']['save_dir']}/{event_name_all[s].split('/')[-1][:-3]}_"
                          f"p{int(patches_idx[s])}_climate.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
