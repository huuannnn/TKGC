import numpy as np
import torch
from tqdm import tqdm
import utils
from core import TKGDataLoader

# Cache for converted tensors to avoid repeated numpy→torch conversion
_total_data_cache = {}

def execute_valid(args, total_data, model,
                  data,
                  s_history, o_history,
                  s_label, o_label,
                  s_frequency, o_frequency):
    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []
    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []
    total_loss2 = 0
    total_loss3 = 0
    batch_count = 0
    
    device = next(model.parameters()).device
    
    # Use cached tensor to avoid repeated conversion
    cache_key = (id(total_data), str(device))
    if cache_key not in _total_data_cache:
        _total_data_cache[cache_key] = torch.from_numpy(total_data).to(device)
    total_data_tensor = _total_data_cache[cache_key]
    
    valid_loader = TKGDataLoader(data, s_history, o_history, 
                                 s_label, o_label, 
                                 s_frequency, o_frequency, 
                                 args.batch_size)
    
    pbar = tqdm(valid_loader, desc="Validating", unit='batch')
    for batch_data in pbar:
        batch_data = [
            batch_data[0].to(device),  # quadruples
            batch_data[1],  # s_history
            batch_data[2],  # o_history
            batch_data[3].to(device),  # s_label
            batch_data[4].to(device),  # o_label
            batch_data[5].to(device),  # s_frequency
            batch_data[6].to(device)   # o_frequency
        ]

        with torch.no_grad():
            _, _, _, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, 'Valid', total_data_tensor)

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3
            
            total_loss2 += cur_loss2.item() if hasattr(cur_loss2, 'item') else cur_loss2
            total_loss3 += cur_loss3.item() if hasattr(cur_loss3, 'item') else cur_loss3
            batch_count += 1
    
    avg_loss = (total_loss2 + total_loss3) / 2.0 / batch_count if batch_count > 0 else 0
    return s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3, avg_loss
