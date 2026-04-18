import os
import numpy as np
import torch
import argparse


def convert_numeric_strings(config):
    for key, value in config.items():
        if not isinstance(value, str):
            continue
        
        # Try to convert to number
        try:
            # Try float first (handles 1e-5, decimals, etc.)
            float_val = float(value)
            # If no decimal point and no scientific notation, make it int
            if '.' not in value and 'e' not in value.lower():
                config[key] = int(float_val)
            else:
                config[key] = float_val
        except ValueError:
            # Not a number, keep as string
            pass
    
    return config


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)




def isListEmpty(inList):
    if isinstance(inList, list):
        return all(map(isListEmpty, inList))
    return False




def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")


def get_gpu_memory_info(device='cuda:0'):
    """Get GPU memory information.
    
    Args:
        device: Device name (e.g., 'cuda:0')
        
    Returns:
        Dict with memory info in GB, or None if unavailable
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        # Get device index from string
        device_idx = int(device.split(':')[1]) if ':' in device else 0
        
        # Clear cache to get accurate readings
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_idx)
        
        reserved = torch.cuda.memory_reserved(device_idx) / 1024 ** 3  # GB
        allocated = torch.cuda.memory_allocated(device_idx) / 1024 ** 3  # GB
        peak = torch.cuda.max_memory_allocated(device_idx) / 1024 ** 3  # GB
        
        # Get total GPU memory
        total = torch.cuda.get_device_properties(device_idx).total_memory / 1024 ** 3  # GB
        
        return {
            'total': total,
            'reserved': reserved,
            'allocated': allocated,
            'peak': peak,
            'free': total - reserved
        }
    except Exception as e:
        return None


def calculate_model_params(model):
    """Calculate total parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def write2file_to_logger(s_ranks, o_ranks, all_ranks, logger):
    """Write evaluation results to logger without console output.
    
    Args:
        s_ranks: Subject ranks
        o_ranks: Object ranks
        all_ranks: All ranks
        logger: Logger object to write to
        
    Returns:
        MRR value
    """
    s_ranks = np.asarray(s_ranks)
    s_mr_lk = np.mean(s_ranks)
    s_mrr_lk = np.mean(1.0 / s_ranks)

    logger.write("    Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
    logger.write("    Subject test MR (lk): {:.6f}".format(s_mr_lk))
    for hit in [1, 3, 10]:
        avg_count_sub_lk = np.mean((s_ranks <= hit))
        logger.write("    Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))

    o_ranks = np.asarray(o_ranks)
    o_mr_lk = np.mean(o_ranks)
    o_mrr_lk = np.mean(1.0 / o_ranks)

    logger.write("    Object test MRR (lk): {:.6f}".format(o_mrr_lk))
    logger.write("    Object test MR (lk): {:.6f}".format(o_mr_lk))
    for hit in [1, 3, 10]:
        avg_count_obj_lk = np.mean((o_ranks <= hit))
        logger.write("    Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk))

    all_ranks = np.asarray(all_ranks)
    all_mr_lk = np.mean(all_ranks)
    all_mrr_lk = np.mean(1.0 / all_ranks)

    logger.write("    ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
    logger.write("    ALL test MR (lk): {:.6f}".format(all_mr_lk))
    for hit in [1, 3, 10]:
        avg_count_all_lk = np.mean((all_ranks <= hit))
        logger.write("    ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk))
    
    return all_mrr_lk
