import os
import time
import csv
import torch
from tqdm import tqdm
from dataset import TKGDataLoader
from training.logger import Logger
import utils
from evaluators import valid


class Trainer:

    def __init__(self, model, optimizer, training_config, dataset, num_relations, num_nodes, num_t,
                 save_dir='SAVE', dataset_name='YAGO', use_cuda=True, device=None):
        self.model = model
        self.optimizer = optimizer
        self.tkg_dataset = dataset
        self.num_relations = num_relations
        self.num_nodes = num_nodes
        self.num_t = num_t
        self.use_cuda = use_cuda
        self.device = device
        self.dataset_name = dataset_name
        self.gpu_id = training_config.get('gpu_id', '0')

        self.device_str = f'cuda:{self.gpu_id}' if use_cuda else 'cpu'

        # Store training config parameters
        self.batch_size = training_config.get('batch_size', 1024)
        self.max_epochs = training_config.get('max_epochs', 30)
        self.valid_epochs = training_config.get('valid_epochs', 5)

        self.logger = Logger(save_dir, dataset_name)
        self.main_dir = self.logger.get_log_dir()
        self.model_path = os.path.join(self.main_dir, 'models')

        os.makedirs(self.model_path, exist_ok=True)

        # Setup CSV metrics logging
        self.metrics_file = os.path.join(self.main_dir, 'metrics.csv')
        self._metrics_csv_f = open(self.metrics_file, 'w', newline='')
        self._metrics_writer = csv.writer(self._metrics_csv_f)
        self._metrics_writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
        self._metrics_csv_f.flush()

    def __del__(self):
        """Đảm bảo đóng file handle khi object bị hủy."""
        if hasattr(self, '_metrics_csv_f') and self._metrics_csv_f:
            self._metrics_csv_f.close()

    def _move_batch_to_device(self, batch_data):
        return [
            batch_data[0].to(self.device),  # quadruples
            batch_data[1],                   # s_history
            batch_data[2],                   # o_history
            batch_data[3].to(self.device),  # s_label
            batch_data[4].to(self.device),  # o_label
            batch_data[5].to(self.device),  # s_frequency
            batch_data[6].to(self.device)   # o_frequency
        ]

    def _log_metrics(self, epoch, train_loss, val_loss=None):
        self._metrics_writer.writerow([
            epoch,
            f'{train_loss:.6f}',
            f'{val_loss:.6f}' if val_loss is not None else ''
        ])
        self._metrics_csv_f.flush()

    def log_config(self, model_config, training_config, system_config):
        self.logger.write("Model Configuration:")
        for key, value in sorted(model_config.items()):
            self.logger.write(f"  {key:20s} = {value}")

        self.logger.write("Training Configuration:")
        for key, value in sorted(training_config.items()):
            self.logger.write(f"  {key:20s} = {value}")

        self.logger.write("System Configuration:")
        for key, value in sorted(system_config.items()):
            self.logger.write(f"  {key:20s} = {value}")
        self.logger.write("")

    def log_model_info(self):
        self.logger.write("Model Information:")
        total_params, trainable_params = utils.calculate_model_params(self.model)
        self.logger.write(f"  Total Parameters: {total_params:,}")
        self.logger.write(f"  Trainable Parameters: {trainable_params:,}")
        mem_info = utils.get_gpu_memory_info(self.device_str)
        if mem_info:
            self.logger.write(f"  GPU Memory Reserved:  {mem_info['reserved']:.2f} GB")
            self.logger.write(f"  GPU Memory Allocated: {mem_info['allocated']:.2f} GB")
            self.logger.write(f"  GPU Memory Peak:      {mem_info['peak']:.2f} GB")

    def train(self):
        best_mrr = 0

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

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            self.logger.write(f"Epoch {epoch}/{self.max_epochs}")

            loss_epoch = 0
            batch_count = 0
            time_begin = time.time()

            pbar = tqdm(train_loader, total=len(train_loader),
                        desc=f"Training Epoch {epoch}", unit='batch')

            for batch_data in pbar:
                batch_data = self._move_batch_to_device(batch_data)

                # giải phóng memory tốt hơn so với zero_grad() mặc định
                self.optimizer.zero_grad(set_to_none=True)

                loss = self.model(batch_data, 'Training')
                if loss is None:
                    continue

                loss_item = loss.item()
                loss.backward()
                self.optimizer.step()

                loss_epoch += loss_item
                batch_count += 1
                avg_loss_batch = loss_epoch / batch_count
                pbar.set_postfix({
                    'loss': f'{loss_item:.4f}',
                    'avg_loss': f'{avg_loss_batch:.4f}'
                })

            epoch_time = time.time() - time_begin
            avg_loss = loss_epoch / batch_count if batch_count > 0 else 0
            self.logger.write(
                f"[TRAIN] Epoch {epoch}: Loss = {avg_loss:.6f} (Time: {epoch_time:.2f}s)"
            )

            if self.use_cuda:
                mem_info = utils.get_gpu_memory_info(self.device_str)
                if mem_info:
                    self.logger.write(
                        f"[VRAM] Peak: {mem_info['peak']:.2f}GB | Reserved: {mem_info['reserved']:.2f}GB"
                    )

            if epoch % self.valid_epochs == 0 and self.dataset_name != 'ICEWS14T':
                best_mrr, val_loss = self._validate(epoch, best_mrr)
                self._log_metrics(epoch, avg_loss, val_loss)
            else:
                self._log_metrics(epoch, avg_loss)

        self.logger.write("Training completed!")
        # Đóng file CSV sau khi training xong
        self._metrics_csv_f.close()

        return self.model_path

    def get_logger(self):
        return self.logger

    def _validate(self, epoch, best_mrr):
        self.logger.write(f"[VALIDATION] Epoch {epoch}")
        # Convert total_data to tensor on device
        total_data_tensor = torch.from_numpy(self.tkg_dataset.total_data).to(self.device)
        s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3, val_loss = valid.execute_valid(
            self.batch_size,
            total_data_tensor,
            self.model,
            self.tkg_dataset.dev_data,
            self.tkg_dataset.dev_s_history,
            self.tkg_dataset.dev_o_history,
            self.tkg_dataset.dev_s_label,
            self.tkg_dataset.dev_o_label,
            self.tkg_dataset.dev_s_frequency,
            self.tkg_dataset.dev_o_frequency
        )

        self.logger.write("  [No Oracle Filtering]")
        raw_mrr = utils.write2file_to_logger(s_ranks2, o_ranks2, all_ranks2, self.logger)
        self.logger.write("  [Ground Truth Oracle]")
        oracle_mrr = utils.write2file_to_logger(s_ranks3, o_ranks3, all_ranks3, self.logger)

        if oracle_mrr > best_mrr:
            best_mrr = oracle_mrr
            best_model_path = os.path.join(self.model_path, f'{self.dataset_name}_best.pth')
            torch.save(self.model.state_dict(), best_model_path)
            self.logger.write(f"  Model improved! Best MRR = {oracle_mrr:.6f}")
        else:
            self.logger.write(f"  No improvement (Best: {best_mrr:.6f} | Current: {oracle_mrr:.6f})")

        return best_mrr, val_loss