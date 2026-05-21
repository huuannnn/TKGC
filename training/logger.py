import csv
import logging
import os
import utils


class Logger:
    def __init__(self, save_dir, dataset_name, version='version'):
        log_id = self._get_next_id(save_dir, version, dataset_name)
        
        folder_name = f"{version}_{log_id}_{dataset_name}"
        self.log_dir = os.path.join(save_dir, folder_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, 'training.log')
        
        self.logger = logging.getLogger(self.log_file)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.handlers = []
        
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)

        self.metrics_file = os.path.join(self.log_dir, 'metrics.csv')
        self._metrics_csv_f = open(self.metrics_file, 'w', newline='')
        self._metrics_writer = csv.writer(self._metrics_csv_f)
        self._metrics_writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
        self._metrics_csv_f.flush()
    
    def get_log_dir(self):
        return self.log_dir
    
    def _get_next_id(self, log_dir, version, dataset_name):
        prefix = f"{version}_"
        suffix = f"_{dataset_name}"
        
        max_id = 0
        if os.path.exists(log_dir):
            for foldername in os.listdir(log_dir):
                if foldername.startswith(prefix) and foldername.endswith(suffix):
                    try:
                        id_str = foldername[len(prefix):-len(suffix)]
                        id_num = int(id_str)
                        max_id = max(max_id, id_num)
                    except (ValueError, IndexError):
                        pass
        
        return max_id + 1
    
    def write(self, message):
        print(message)
        self.logger.info(message)

    def log_metrics(self, epoch, train_loss, val_loss=None):
        self._metrics_writer.writerow([
            epoch,
            f'{train_loss:.6f}',
            f'{val_loss:.6f}' if val_loss is not None else ''
        ])
        self._metrics_csv_f.flush()

    def log_config(self, model_config, training_config, system_config):
        self.write("Model Configuration:")
        for key, value in sorted(model_config.items()):
            self.write(f"  {key:20s} = {value}")

        self.write("Training Configuration:")
        for key, value in sorted(training_config.items()):
            self.write(f"  {key:20s} = {value}")

        self.write("System Configuration:")
        for key, value in sorted(system_config.items()):
            self.write(f"  {key:20s} = {value}")
        self.write("")

    def log_model_info(self, model, device_str):
        self.write("Model Information:")
        total_params, trainable_params = utils.calculate_model_params(model)
        self.write(f"  Total Parameters: {total_params:,}")
        self.write(f"  Trainable Parameters: {trainable_params:,}")
        mem_info = utils.get_gpu_memory_info(device_str)
        if mem_info:
            self.write(f"  GPU Memory Reserved:  {mem_info['reserved']:.2f} GB")
            self.write(f"  GPU Memory Allocated: {mem_info['allocated']:.2f} GB")
            self.write(f"  GPU Memory Peak:      {mem_info['peak']:.2f} GB")

    def log_epoch_start(self, epoch, max_epochs):
        self.write(f"Epoch {epoch}/{max_epochs}")

    def log_train_epoch(self, epoch, avg_loss, epoch_time):
        self.write(
            f"[TRAIN] Epoch {epoch}: Loss = {avg_loss:.6f} (Time: {epoch_time:.2f}s)"
        )

    def log_gpu_memory(self, device_str):
        mem_info = utils.get_gpu_memory_info(device_str)
        if mem_info:
            self.write(
                f"[VRAM] Peak: {mem_info['peak']:.2f}GB | Reserved: {mem_info['reserved']:.2f}GB"
            )

    def log_validation_start(self, epoch):
        self.write(f"[VALIDATION] Epoch {epoch}")

    def log_validation_metrics(self, s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3):
        self.write("  [No Oracle Filtering]")
        raw_mrr = utils.write2file_to_logger(s_ranks2, o_ranks2, all_ranks2, self)
        self.write("  [Ground Truth Oracle]")
        oracle_mrr = utils.write2file_to_logger(s_ranks3, o_ranks3, all_ranks3, self)
        return raw_mrr, oracle_mrr

    def log_model_improved(self, oracle_mrr):
        self.write(f"  Model improved! Best MRR = {oracle_mrr:.6f}")

    def log_model_not_improved(self, best_mrr, oracle_mrr):
        self.write(f"  No improvement (Best: {best_mrr:.6f} | Current: {oracle_mrr:.6f})")

    def log_training_completed(self):
        self.write("Training completed!")
    
    def close(self):
        if hasattr(self, '_metrics_csv_f') and self._metrics_csv_f:
            self._metrics_csv_f.close()
            self._metrics_csv_f = None

        for handler in self.logger.handlers:
            handler.close()