import os
import time
import shutil
import csv
from datetime import datetime
import torch
from tqdm import tqdm
from core.dataset import TKGDataLoader
from core.logger import Logger
import utils
import valid


class Trainer:
    
    def __init__(self, model, optimizer, args, dataset, num_relations, num_nodes, num_t, 
                 use_cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.dataset = dataset
        self.num_relations = num_relations
        self.num_nodes = num_nodes
        self.num_t = num_t
        self.use_cuda = use_cuda
        
        # Create output directory
        now = datetime.now()
        dt_string = args.description + now.strftime("%d-%m-%Y,%H-%M-%S") + \
                    args.dataset + '-EPOCH' + str(args.max_epochs)
        self.main_dir = os.path.join(args.save_dir, dt_string)
        self.model_path = os.path.join(self.main_dir, 'models')
        
        os.makedirs(self.main_dir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        
        # Setup single unified logger
        self.logger = Logger(os.path.join(self.main_dir, 'training.log'))
        
        # Setup CSV metrics logging
        self.metrics_file = os.path.join(self.main_dir, 'metrics.csv')
        self._init_metrics_csv()

    def _move_batch_to_device(self, batch_data):
        return [
            batch_data[0].to(self.args.device),  # quadruples
            batch_data[1],  # s_history
            batch_data[2],  # o_history
            batch_data[3].to(self.args.device),  # s_label
            batch_data[4].to(self.args.device),  # o_label
            batch_data[5].to(self.args.device),  # s_frequency
            batch_data[6].to(self.args.device)   # o_frequency
        ]
    
    def _init_metrics_csv(self):
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
    
    def _log_metrics(self, epoch, train_loss, val_loss=None):
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{train_loss:.6f}',
                f'{val_loss:.6f}' if val_loss is not None else ''
            ])
    

    
    def log_config(self, config_default_path=None, config_dataset_path=None):
        if config_default_path:
            self.logger.write(f"Loaded config from: {config_default_path}")
        if config_dataset_path:
            self.logger.write(f"Loaded config from: {config_dataset_path}")
        for key, value in sorted(vars(self.args).items()):
            self.logger.write(f"  {key:20s} = {value}")
        self.logger.write("")
        if config_default_path and os.path.exists(config_default_path):
            dest_default = os.path.join(self.main_dir, os.path.basename(config_default_path))
            shutil.copy2(config_default_path, dest_default)
            self.logger.write(f"Saved config to: {dest_default}")
        
        if config_dataset_path and os.path.exists(config_dataset_path):
            dest_dataset = os.path.join(self.main_dir, os.path.basename(config_dataset_path))
            shutil.copy2(config_dataset_path, dest_dataset)
            self.logger.write(f"Saved config to: {dest_dataset}")
    
    def log_model_info(self):
        self.logger.write("Model Information:")
        total_params, trainable_params = utils.calculate_model_params(self.model)
        self.logger.write(f"  Total Parameters: {total_params:,}")
        self.logger.write(f"  Trainable Parameters: {trainable_params:,}")
        mem_info = utils.get_gpu_memory_info(f'cuda:{self.args.gpu_id}' if self.use_cuda else 'cpu')
        if mem_info:
            self.logger.write(f"  GPU Memory Reserved: {mem_info['reserved']:.2f} GB")
            self.logger.write(f"  GPU Memory Allocated: {mem_info['allocated']:.2f} GB")
            self.logger.write(f"  GPU Memory Peak: {mem_info['peak']:.2f} GB")
    
    def train(self):
        best_mrr = 0
        
        for epoch in range(1, self.args.max_epochs + 1):
            self.model.train()
            self.logger.write(f"Epoch {epoch}/{self.args.max_epochs}")
            
            loss_epoch = 0
            batch_count = 0
            time_begin = time.time()
            train_loader = TKGDataLoader(
                self.dataset.train_data,
                self.dataset.train_s_history,
                self.dataset.train_o_history,
                self.dataset.train_s_label,
                self.dataset.train_o_label,
                self.dataset.train_s_frequency,
                self.dataset.train_o_frequency,
                self.args.batch_size
            )
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f"Training Epoch {epoch}", unit='batch')
            
            for batch_idx, batch_data in pbar:
                batch_data = self._move_batch_to_device(batch_data)
                loss = self.model(batch_data, 'Training')
                if loss is None:
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                loss_item = loss.item()
                loss_epoch += loss_item
                batch_count += 1
                pbar.set_postfix({
                    'loss': f'{loss_item:.4f}',
                    'avg_loss': f'{loss_epoch/batch_count:.4f}'
                })
            pbar.close()
            epoch_time = time.time() - time_begin
            avg_loss = loss_epoch / batch_count if batch_count > 0 else 0
            self.logger.write(
                f"[TRAIN] Epoch {epoch}: Loss = {avg_loss:.6f} (Time: {epoch_time:.2f}s)"
            )
            if self.use_cuda:
                mem_info = utils.get_gpu_memory_info(f'cuda:{self.args.gpu_id}')
                if mem_info:
                    self.logger.write(
                        f"[VRAM] Peak: {mem_info['peak']:.2f}GB | Reserved: {mem_info['reserved']:.2f}GB"
                    )
            if epoch % self.args.valid_epochs == 0 and self.args.dataset != 'ICEWS14T':
                best_mrr, val_loss = self._validate(epoch, best_mrr)
                self._log_metrics(epoch, avg_loss, val_loss)
            else:
                self._log_metrics(epoch, avg_loss)
        
        self.logger.write("Training completed!")
        
        return self.model_path
    
    def get_logger(self):
        return self.logger
    

    def _validate(self, epoch, best_mrr):
        self.logger.write(f"[VALIDATION] Epoch {epoch}")
        s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3, val_loss = valid.execute_valid(
            self.args,
            self.dataset.total_data,
            self.model,
            self.dataset.dev_data,
            self.dataset.dev_s_history,
            self.dataset.dev_o_history,
            self.dataset.dev_s_label,
            self.dataset.dev_o_label,
            self.dataset.dev_s_frequency,
            self.dataset.dev_o_frequency
        )

        self.logger.write("  [No Oracle Filtering]")
        raw_mrr = utils.write2file_to_logger(s_ranks2, o_ranks2, all_ranks2, self.logger)
        self.logger.write("  [Ground Truth Oracle]")
        oracle_mrr = utils.write2file_to_logger(s_ranks3, o_ranks3, all_ranks3, self.logger)
        if oracle_mrr > best_mrr:
            best_mrr = oracle_mrr
            best_model_path = os.path.join(self.model_path, f'{self.args.dataset}_best.pth')
            torch.save(self.model, best_model_path)
            self.logger.write(f"  Model improved! Best MRR = {oracle_mrr:.6f}")
        else:
            self.logger.write(f"  No improvement (Best: {best_mrr:.6f} | Current: {oracle_mrr:.6f})")
        return best_mrr, val_loss
