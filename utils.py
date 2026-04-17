import os
import numpy as np
import torch
import argparse


def convert_numeric_strings(config):
    """Convert numeric strings to proper types (e.g., '1e-5' -> 0.00001).
    
    Args:
        config (dict): Configuration dictionary with potentially string numeric values
        
    Returns:
        dict: Configuration with numeric strings converted to int/float
    """
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


def make_batch(a, b, c, d, e, f, g, batch_size, valid1=None, valid2=None):
    if valid1 is None and valid2 is None:
        for i in range(0, len(a), batch_size):
            yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
                   d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size]]
    else:
        for i in range(0, len(a), batch_size):
            yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
                   d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size],
                   valid1[i:i + batch_size], valid2[i:i + batch_size]]


def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor.cpu()


def isListEmpty(inList):
    if isinstance(inList, list):
        return all(map(isListEmpty, inList))
    return False


def get_sorted_s_r_embed_limit(s_hist, s, r, ent_embeds, limit):
    s_hist_len = to_device(torch.LongTensor(list(map(len, s_hist))))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]
    s_len_non_zero = torch.where(s_len_non_zero > limit, to_device(torch.tensor(limit)), s_len_non_zero)

    s_hist_sorted = []
    for idx in s_idx[:num_non_zero]:
        s_hist_sorted.append(s_hist[idx.item()])

    flat_s = []
    len_s = []

    for hist in s_hist_sorted:
        for neighs in hist[-limit:]:
            len_s.append(len(neighs))
            for neigh in neighs:
                flat_s.append(neigh[1])
    s_tem = s[s_idx]
    r_tem = r[s_idx]

    embeds = ent_embeds[to_device(torch.LongTensor(flat_s))]
    embeds_split = torch.split(embeds, len_s)
    return s_idx, s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split


def get_sorted_s_r_embed(s_hist, s, r, ent_embeds):
    s_hist_len = to_device(torch.LongTensor(list(map(len, s_hist))))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]

    s_hist_sorted = []
    for idx in s_idx[:num_non_zero]:
        s_hist_sorted.append(s_hist[idx.item()])

    flat_s = []
    len_s = []

    for hist in s_hist_sorted:
        for neighs in hist:
            len_s.append(len(neighs))
            for neigh in neighs:
                flat_s.append(neigh[1])
    s_tem = s[s_idx]
    r_tem = r[s_idx]

    embeds = ent_embeds[to_device(torch.LongTensor(flat_s))]
    embeds_split = torch.split(embeds, len_s)
    """
    s_idx: id of descending by length in original list.  1 * batch
    s_len_non_zero: number of events having history  any
    s_tem: sorted s by length  batch
    r_tem: sorted r by length  batch
    embeds: event->history->neighbor
    lens_s: event->history_neighbor length
    embeds_split split by history neighbor length
    s_hist_dt_sorted: history interval sorted by history length without non
    """
    return s_idx, s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split


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


def write2file(s_ranks, o_ranks, all_ranks, file_test):
    s_ranks = np.asarray(s_ranks)
    s_mr_lk = np.mean(s_ranks)
    s_mrr_lk = np.mean(1.0 / s_ranks)

    # print("Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
    # print("Subject test MR (lk): {:.6f}".format(s_mr_lk))
    if file_test is not None:
        file_test.write("Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
        file_test.write("Subject test MR (lk): {:.6f}".format(s_mr_lk))
    for hit in [1, 3, 10]:
        avg_count_sub_lk = np.mean((s_ranks <= hit))
        # print("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))
        if file_test is not None:
            file_test.write("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))

    o_ranks = np.asarray(o_ranks)
    o_mr_lk = np.mean(o_ranks)
    o_mrr_lk = np.mean(1.0 / o_ranks)

    # print("Object test MRR (lk): {:.6f}".format(o_mrr_lk))
    # print("Object test MR (lk): {:.6f}".format(o_mr_lk))
    if file_test is not None:
        file_test.write("Object test MRR (lk): {:.6f}".format(o_mrr_lk))
        file_test.write("Object test MR (lk): {:.6f}".format(o_mr_lk))
    for hit in [1, 3, 10]:
        avg_count_obj_lk = np.mean((o_ranks <= hit))
        # print("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk))
        if file_test is not None:
            file_test.write("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk))

    all_ranks = np.asarray(all_ranks)
    all_mr_lk = np.mean(all_ranks)
    all_mrr_lk = np.mean(1.0 / all_ranks)

    # print("ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
    # print("ALL test MR (lk): {:.6f}".format(all_mr_lk))
    if file_test is not None:
        file_test.write("ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
        file_test.write("ALL test MR (lk): {:.6f}".format(all_mr_lk))
    for hit in [1, 3, 10]:
        avg_count_all_lk = np.mean((all_ranks <= hit))
        # print("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk))
        if file_test is not None:
            file_test.write("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk))
    return all_mrr_lk


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
