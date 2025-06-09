import math
import logging
from tqdm import tqdm
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
logger = logging.getLogger(__name__)
import h5py
import torch.nn.functional as F  # Import functional for padding
import wandb
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os



class TrainerConfig:
    k=10
    max_epochs = 10
    batch_size = 64
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0 
    rtg_dim=1
    n_embd=128
    query_result_dim=1
    query_dim=10
    block_size=128
    embd_pdrop = 0.1
    n_layer=6
    n_head=8
    attn_pdrop=0.1
    resid_pdrop=0.1
    pad_scalar_val=-100
    pad_vec_val=0


    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)




class Trainer:
    
    def __init__(self, model, dataloader, device, rank, config):

        
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.rank = rank
        self.config = config

        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.dataset_path="atari/data_6e6.h5"


        # Check if CUDA (or ROCm for AMD GPUs) is available
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load dataset from .h5 file
        #self.load_dataset()

        # Check if the model is being moved to the GPU correctly
        # model = model.to(device)
        # print(f"Model is on device: {next(model.parameters()).device}")
        # self.optimizer = model.configure_optimizers(config)
    

 






    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        #raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = model.configure_optimizers(config)

        self.tokens = 0  
        def run_epoch(mode,epoch_num=0):
            is_train = mode == 'train'
            model.train(is_train)
            losses = []
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch_num+1}")            
            total_loss = 0.0
            for q, r, rtg, mask_lengths in self.dataloader:
                # Move data to GPU if available
                q, r, rtg, mask_lengths = q.to(self.device), r.to(self.device), rtg.to(self.device), mask_lengths.to(self.device)
                
                with torch.set_grad_enabled(True):
                    _, loss = model(mask_lengths,rtg, r, q, q)
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    # for name, param in model.named_parameters():
                    #     # if param.grad is not None:
                    #     #     print(f"{name} gradient sum: {param.grad.sum()}")
                    #     grad_norm = param.grad.norm()
                    #     print(f"Gradient norm for {name}: {grad_norm}")
                    # print("new")
                    # print("\n") 
                    # time.sleep(1)
                    
                    
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    

                    if self.rank==0:
                        # Log loss and gradients to WandB
                        wandb.log({
                            "loss": loss.item(),
                            "epoch": epoch
                        })

                                # Log gradient norms
                        grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None]), 2)
                        wandb.log({"gradient_norm": grad_norm.item()})
                        
                    
                    
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    #report progress
                    pbar.set_description(f"Epoch {epoch_num+1} Loss: {loss.item():.5f}, LR: {lr:e}")
                    pbar.update(1)
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss
            
            avg_loss = float(np.mean(losses))
            logger.info("Epoch %d - Avg Loss: %f", epoch + 1, avg_loss)

            if config.ckpt_path is not None:
                self.save_checkpoint()
                print("check_point saved")


        # Initialize wandb
        #wandb.init(project="DT", mode="disabled")


        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)

        if self.rank==0:
            wandb.finish()

