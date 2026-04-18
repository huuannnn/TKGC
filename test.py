import argparse
import numpy as np
import torch
import pickle
import time
import datetime
import os
import random
from tqdm import tqdm
from cenet_model import CENET
from core import TKGDataLoader


def execute_test(args, total_data, model,
                 data,
                 s_history, o_history,
                 s_label, o_label,
                 s_frequency, o_frequency):
    s_ranks1 = []
    o_ranks1 = []
    all_ranks1 = []

    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []

    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []
    device = args.device if hasattr(args, 'device') else torch.device('cpu')
    total_data = torch.from_numpy(total_data).to(device)
    
    test_loader = TKGDataLoader(data, s_history, o_history, 
                                s_label, o_label, 
                                s_frequency, o_frequency, 
                                args.batch_size)
    
    # Testing with progress bar
    pbar = tqdm(test_loader, desc="Testing", unit='batch')
    for batch_data in pbar:
        # Move batch data to device
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
            sub_rank1, obj_rank1, cur_loss1, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, 'Test', total_data)

            s_ranks1 += sub_rank1
            o_ranks1 += obj_rank1
            tmp1 = sub_rank1 + obj_rank1
            all_ranks1 += tmp1

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3
    
    pbar.close()
    return s_ranks1, o_ranks1, all_ranks1, \
           s_ranks2, o_ranks2, all_ranks2, \
           s_ranks3, o_ranks3, all_ranks3
