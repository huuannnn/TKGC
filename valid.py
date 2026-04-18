import argparse
import numpy as np
import torch
import pickle
import time
import datetime
import os
import random
from tqdm import tqdm
import utils
from cenet_model import CENET
from core import TKGDataLoader


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
    device = next(model.parameters()).device
    total_data = torch.from_numpy(total_data).to(device)
    
    valid_loader = TKGDataLoader(data, s_history, o_history, 
                                 s_label, o_label, 
                                 s_frequency, o_frequency, 
                                 args.batch_size,
                                 model=model)
    
    # Validation with progress bar
    pbar = tqdm(valid_loader, desc="Validating", unit='batch')
    for batch_data in pbar:

        with torch.no_grad():
            _, _, _, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, 'Valid', total_data)

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3
    
    pbar.close()
    return s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3
