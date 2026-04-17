import argparse
import os
import time
import yaml
import shutil
from types import SimpleNamespace

import numpy as np
import torch

import utils
import test
import valid
from cenet_model import CENET
from core import TKGDataset, Trainer, OracleTrainer, Logger


def main_portal(args, config_default_path=None, config_dataset_path=None):
    """Main training pipeline."""
    # Set seed for reproducibility from config (if enabled)
    if hasattr(args, 'use_seed') and args.use_seed:
        seed = args.seed if hasattr(args, 'seed') else 987
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Load dataset
    dataset = TKGDataset(args.dataset)
    num_nodes, num_rels, num_t = dataset.num_nodes, dataset.num_rels, dataset.num_t
    
    # Initialize trainer for logging
    trainer = None
    test_logger = None
    
    # Training phase
    if not args.only_eva and not args.only_oracle:
        model = CENET(num_nodes, num_rels, num_t, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.use_cuda:
            model = model.cuda()
        
        trainer = Trainer(model, optimizer, args, dataset, num_rels, num_nodes, num_t, args.use_cuda)
        trainer.log_config(config_default_path, config_dataset_path)
        trainer.log_model_info()
        model_path = trainer.train()
        main_dirName = os.path.dirname(model_path)
        test_logger = trainer.get_logger()
    else:
        # Load model from existing training
        main_dirName = os.path.join(args.save_dir, args.model_dir)
        model_path = os.path.join(main_dirName, 'models')
        # Create logger for evaluation only
        test_logger = Logger(os.path.join(main_dirName, 'training.log'))
    
    # Oracle training phase
    model = torch.load(os.path.join(model_path, f'{args.dataset}_best.pth'), weights_only=False)
    oracle_trainer = OracleTrainer(model, args, dataset, model_path, args.use_cuda, logger=test_logger)
    oracle_trainer.train()
    torch.save(model, os.path.join(model_path, f'{args.dataset}_best.pth'))
    
    # Evaluation phase
    if args.only_eva:
        main_dirName = os.path.join(args.save_dir, args.model_dir)
        model_path = os.path.join(main_dirName, 'models')
    
    test_logger.write("[TEST PHASE]")
    test_logger.write("Testing starts...")
    
    time_begin = time.time()
    
    model = torch.load(os.path.join(model_path, f'{args.dataset}_best.pth'), weights_only=False)
    model.eval()
    model.args = args
    
    # Run test
    s_ranks1, o_ranks1, all_ranks1, _, _, _, _, _, _ = test.execute_test(
        args, dataset.total_data, model, dataset.test_data,
        dataset.test_s_history, dataset.test_o_history,
        dataset.test_s_label, dataset.test_o_label,
        dataset.test_s_frequency, dataset.test_o_frequency
    )
    
    test_time = time.time() - time_begin
    test_logger.write("Test Results (Oracle):")
    utils.write2file_to_logger(s_ranks1, o_ranks1, all_ranks1, test_logger)
    test_logger.write(f"Testing completed! (Time: {test_time:.2f}s)")
    test_logger.close()


def load_config(config_file):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            # print(f"Loaded config from: {config_file}")
            return config if config is not None else {}
    except FileNotFoundError:
        # print(f"Config file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        # print(f"Error parsing YAML file {config_file}: {e}")
        raise


def merge_configs(default_config, dataset_config):
    """Merge dataset config into default config (dataset overrides defaults)."""
    merged = default_config.copy()
    merged.update(dataset_config)
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CENET - Temporal Knowledge Graph Embedding')
    
    parser.add_argument("--config_default", type=str, default='configs/default.yaml', 
                        help="Path to default config file")
    parser.add_argument("--config_dataset", type=str, default='configs/YAGO.yaml', 
                        help="Path to dataset-specific config file")
    
    args = parser.parse_args()
    
    # Load configs using PyYAML
    # print("\n" + "="*60)
    # print("Loading configuration...")
    # print("="*60)
    default_config = load_config(args.config_default)
    dataset_config = load_config(args.config_dataset)
    
    # Merge configs (dataset overrides defaults)
    final_config = merge_configs(default_config, dataset_config)
    
    # Convert numeric strings to proper types (e.g., '1e-5' -> float)
    final_config = utils.convert_numeric_strings(final_config)
    args_main = SimpleNamespace(**final_config)
    
    # Create save directory
    os.makedirs(args_main.save_dir, exist_ok=True)
    
    # Setup GPU/CPU based on configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_main.gpu_id)
    use_cuda = args_main.use_cuda and torch.cuda.is_available()
    args_main.use_cuda = use_cuda  # Update args with actual CUDA availability
    
    # if use_cuda:
    #     print(f"GPU Device: {args_main.gpu_id}")
    # else:
    #     print("CUDA is not available. Using CPU instead.")
    
    # print("\n" + "="*60)
    # print("Configuration Summary:")
    # print("="*60)
    # for key, value in sorted(vars(args_main).items()):
    #     print(f"  {key:20s} = {value}")
    # print("="*60 + "\n")
    
    main_portal(args_main, args.config_default, args.config_dataset)
