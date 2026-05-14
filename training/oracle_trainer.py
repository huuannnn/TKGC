import os
import time
import torch
from tqdm import tqdm
from dataset import TKGDataLoader
from training.logger import Logger


class OracleTrainer:
    def __init__(self, model, training_config, dataset, model_path, dataset_name='YAGO', 
                 use_cuda=True, device=None, logger=None):
        self.model = model
        self.tkg_dataset = dataset
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.device = device
        self.dataset_name = dataset_name
        
        # Extract training parameters
        self.oracle_lr = training_config.get('oracle_lr', 0.001)
        self.weight_decay = training_config.get('weight_decay', 1.0e-05)
        self.oracle_epochs = training_config.get('oracle_epochs', 20)
        self.batch_size = training_config.get('batch_size', 1024)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.oracle_lr,
            weight_decay=self.weight_decay
        )
        self.model.freeze_parameter()
        
        # Setup logger
        if logger is None:
            self.logger = Logger(os.path.dirname(model_path), dataset_name)
        else:
            self.logger = logger
        
        self.logger.write("[ORACLE TRAINING PHASE]")
        self.logger.write("Starting Oracle training...")
    
    def _prepare_batch(self, batch_data):
        if self.use_cuda:
            batch_data[0] = batch_data[0].to(self.device)
            batch_data[3] = batch_data[3].to(self.device)
            batch_data[4] = batch_data[4].to(self.device)
            batch_data[5] = batch_data[5].to(self.device)
            batch_data[6] = batch_data[6].to(self.device)
        
        return batch_data
    
    def train(self):
        """Run oracle training loop."""
        
        for oracle_epoch in range(1, self.oracle_epochs + 1):
            
            total_loss = 0
            batch_count = 0
            time_begin = time.time()
            
            # Create data loader
            train_loader = TKGDataLoader(
                self.tkg_dataset.train_data,
                self.tkg_dataset.train_s_history,
                self.tkg_dataset.train_o_history,
                self.tkg_dataset.train_s_label,
                self.tkg_dataset.train_o_label,
                self.tkg_dataset.train_s_frequency,
                self.tkg_dataset.train_o_frequency,
                self.batch_size
            )
            
            # Oracle training batches with progress bar
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                       desc=f"Oracle Epoch {oracle_epoch}", unit='batch')
            
            for batch_idx, batch_data in pbar:
                # Move batch data to device
                batch_data = self._prepare_batch(batch_data)
                loss = self.model(batch_data, 'Oracle')
                if loss is None:
                    continue
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                loss_item = loss.item()
                total_loss += loss_item
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_item:.4f}',
                    'avg_loss': f'{total_loss/batch_count:.4f}'
                })
            
            pbar.close()
            
            # Epoch summary
            epoch_time = time.time() - time_begin
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            self.logger.write(
                f"[ORACLE] Epoch {oracle_epoch}: Loss = {avg_loss:.6f} (Time: {epoch_time:.2f}s)"
            )
        
        self.logger.write("Oracle training completed!")
