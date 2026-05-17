import argparse
import os
import time
import yaml
import shutil
from types import SimpleNamespace

import numpy as np
import torch

import utils
from evaluators import evaluation, valid
from models import CENET
from dataset import TKGDataset
from training import Trainer, OracleTrainer, Logger


def load_config(config_file):
    """Load config from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            return config if config is not None else {}
    except FileNotFoundError:
        raise
    except yaml.YAMLError as e:
        raise


def merge_configs(default_config, dataset_config):
    """Merge default and dataset configs"""
    merged = default_config.copy()
    merged.update(dataset_config)
    return merged


def build_model(args, num_nodes, num_rels, num_t):
    """Helper: khởi tạo CENET model và đưa lên device."""
    model = CENET(
        num_nodes,
        num_rels,
        num_t,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        lambdax=args.lambdax,
        alpha=args.alpha,
        oracle_mode=args.oracle_mode,
        filtering=args.filtering,
    )
    return model.to(args.device)


def save_model(model, path):
    """Lưu state_dict của model."""
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """Load state_dict vào model đã khởi tạo sẵn."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def main_portal(args, model_config, training_config, system_config):
    """Main training and evaluation pipeline"""

    # Load dataset
    dataset = TKGDataset(args.dataset)
    num_nodes, num_rels, num_t = dataset.num_nodes, dataset.num_rels, dataset.num_t

    best_model_name = f'{args.dataset}_best.pth'
    test_logger = None

    # ------------------------------------------------------------------ #
    # 1. Training phase                                                    #
    # ------------------------------------------------------------------ #
    if not args.only_eva and not args.only_oracle:
        model = build_model(args, num_nodes, num_rels, num_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        trainer = Trainer(
            model, optimizer, training_config, dataset, num_rels, num_nodes, num_t,
            save_dir=args.save_dir,
            dataset_name=args.dataset,
            use_cuda=args.use_cuda,
            device=args.device
        )
        trainer.log_config(model_config, training_config, system_config)
        trainer.log_model_info()
        model_path = trainer.train()          # trả về thư mục chứa best.pth
        main_dirName = os.path.dirname(model_path)
        test_logger = trainer.get_logger()
    else:
        # Load từ training đã có sẵn
        main_dirName = os.path.join(args.save_dir, args.model_dir)
        model_path = os.path.join(main_dirName, 'models')
        logger_instance = Logger(args.save_dir, args.dataset)
        test_logger = logger_instance
        main_dirName = logger_instance.get_log_dir()

    # ------------------------------------------------------------------ #
    # 2. Oracle training phase                                             #
    # ------------------------------------------------------------------ #
    model = build_model(args, num_nodes, num_rels, num_t)
    load_model(model, os.path.join(model_path, best_model_name), args.device)

    oracle_trainer = OracleTrainer(
        model, training_config, dataset, model_path,
        dataset_name=args.dataset,
        use_cuda=args.use_cuda,
        device=args.device,
        logger=test_logger
    )
    oracle_trainer.train()

    # Lưu lại sau oracle training
    save_model(model, os.path.join(model_path, best_model_name))

    # ------------------------------------------------------------------ #
    # 3. Evaluation phase                                                  #
    # ------------------------------------------------------------------ #
    if args.only_eva:
        main_dirName = os.path.join(args.save_dir, args.model_dir)
        model_path = os.path.join(main_dirName, 'models')

    test_logger.write("[TEST PHASE]")
    test_logger.write("Testing starts...")

    time_begin = time.time()

    model = build_model(args, num_nodes, num_rels, num_t)
    load_model(model, os.path.join(model_path, best_model_name), args.device)
    model.eval()
    
    total_data_tensor = torch.from_numpy(dataset.total_data).to(args.device)


    # Run test
    s_ranks1, o_ranks1, all_ranks1, _, _, _, _, _, _, avg_loss = evaluation.execute_test(
        training_config['batch_size'],
        total_data_tensor,
        model,
        dataset.test_data,
        dataset.test_s_history, dataset.test_o_history,
        dataset.test_s_label, dataset.test_o_label,
        dataset.test_s_frequency, dataset.test_o_frequency
    )

    test_time = time.time() - time_begin
    test_logger.write("Test Results (Oracle):")
    utils.write2file_to_logger(s_ranks1, o_ranks1, all_ranks1, test_logger)
    test_logger.write(f"Testing completed! (Time: {test_time:.2f}s)")
    test_logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CENET - Temporal Knowledge Graph Embedding')

    parser.add_argument("--dataset", type=str, default='YAGO',
                        help="Dataset name (e.g., YAGO, ICEWS14, ICEWS18, GDELT, WIKI)")
    parser.add_argument("--config_dir", type=str, default='configs',
                        help="Directory containing config files")

    args = parser.parse_args()

    # Auto-construct config paths
    config_default_path = os.path.join(args.config_dir, 'default.yaml')
    config_dataset_path = os.path.join(args.config_dir, f'{args.dataset}.yaml')

    # Load and merge configs (dataset overrides defaults)
    default_config = load_config(config_default_path)
    dataset_config = load_config(config_dataset_path)
    final_config = merge_configs(default_config, dataset_config)
    final_config['dataset'] = args.dataset

    # Convert numeric strings to proper types (e.g., '1e-5' -> float)
    final_config = utils.convert_numeric_strings(final_config)
    args_main = SimpleNamespace(**final_config)

    # Create save directory
    os.makedirs(args_main.save_dir, exist_ok=True)

    # Setup GPU/CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_main.gpu_id)
    use_cuda = args_main.use_cuda and torch.cuda.is_available()
    args_main.use_cuda = use_cuda
    args_main.device = torch.device(f'cuda:{args_main.gpu_id}' if use_cuda else 'cpu')

    # Set seed
    if args_main.use_seed:
        seed = args_main.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Split config cho từng phase
    model_config = {
        'embedding_dim': args_main.embedding_dim,
        'dropout': args_main.dropout,
        'alpha': args_main.alpha,
        'lambdax': args_main.lambdax,
        'oracle_mode': args_main.oracle_mode,
        'filtering': args_main.filtering,
    }

    training_config = {
        'batch_size': args_main.batch_size,
        'max_epochs': args_main.max_epochs,
        'oracle_epochs': args_main.oracle_epochs,
        'valid_epochs': args_main.valid_epochs,
        'lr': args_main.lr,
        'oracle_lr': args_main.oracle_lr,
        'weight_decay': args_main.weight_decay,
        'use_seed': args_main.use_seed,
        'seed': args_main.seed,
    }

    system_config = {
        'gpu_id': args_main.gpu_id,
        'use_cuda': args_main.use_cuda,
        'only_oracle': args_main.only_oracle,
        'only_eva': args_main.only_eva,
        'save_dir': args_main.save_dir,
        'model_dir': args_main.model_dir,
        'dataset': args_main.dataset,
    }

    main_portal(args_main, model_config, training_config, system_config)