import os
import time
import torch
from tqdm import tqdm
from core.dataset import TKGDataLoader
from core.logger import Logger


class OracleTrainer:
    def __init__(self, model, args, dataset, model_path, use_cuda=True, logger=None):
        self.model = model
        self.args = args
        self.dataset = dataset
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.oracle_lr,
            weight_decay=args.weight_decay
        )
        self.model.freeze_parameter()
        
        # Setup logger
        if logger is None:
            log_file = os.path.join(os.path.dirname(model_path), 'oracle.log')
            self.logger = Logger(log_file)
        else:
            self.logger = logger
        
        self.logger.write("[ORACLE TRAINING PHASE]")
        self.logger.write("Starting Oracle training...")
    
    def _prepare_batch(self, batch_data):
        """Prepare batch data for model."""
        batch_data[0] = torch.from_numpy(batch_data[0])
        batch_data[3] = torch.from_numpy(batch_data[3]).float()
        batch_data[4] = torch.from_numpy(batch_data[4]).float()
        batch_data[5] = torch.from_numpy(batch_data[5]).float()
        batch_data[6] = torch.from_numpy(batch_data[6]).float()
        
        if self.use_cuda:
            batch_data[0] = batch_data[0].cuda()
            batch_data[3] = batch_data[3].cuda()
            batch_data[4] = batch_data[4].cuda()
            batch_data[5] = batch_data[5].cuda()
            batch_data[6] = batch_data[6].cuda()
        
        return batch_data
    
    def train(self):
        """Run oracle training loop."""
        # self.logger.write(f"\n{'='*60}")
        # self.logger.write(f"Oracle Training - {self.args.oracle_epochs} Epochs")
        # self.logger.write(f"{'='*60}\n")
        
        for oracle_epoch in range(1, self.args.oracle_epochs + 1):
            # self.logger.write(f"Oracle Epoch {oracle_epoch}/{self.args.oracle_epochs}")
            
            total_loss = 0
            batch_count = 0
            time_begin = time.time()
            
            # Create data loader
            train_loader = TKGDataLoader(
                self.dataset.train_data,
                self.dataset.train_s_history,
                self.dataset.train_o_history,
                self.dataset.train_s_label,
                self.dataset.train_o_label,
                self.dataset.train_s_frequency,
                self.dataset.train_o_frequency,
                self.args.batch_size,
                model=self.model
            )
            
            # Oracle training batches with progress bar
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                       desc=f"Oracle Epoch {oracle_epoch}", unit='batch')
            
            for batch_idx, batch_data in pbar:
                loss = self.model(batch_data, 'Oracle')
                if loss is None:
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
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
