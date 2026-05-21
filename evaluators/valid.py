import torch
from tqdm import tqdm
from .utils import _move_batch_to_device


def execute_valid(batch_size, total_data_tensor, model, valid_loader, device=None):
    s_ranks2, o_ranks2, all_ranks2 = [], [], []
    s_ranks3, o_ranks3, all_ranks3 = [], [], []
    total_loss = 0.0
    batch_count = 0

    if device is None:
        device = total_data_tensor.device

    model.eval()
    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Validating", unit='batch')
        for batch_data in pbar:
            batch_data = _move_batch_to_device(batch_data, device)

            cur_loss, _, _, \
            sub_rank2, obj_rank2, \
            sub_rank3, obj_rank3, ce_all_acc = model(batch_data, 'Valid', total_data_tensor)

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            all_ranks2 += sub_rank2 + obj_rank2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            all_ranks3 += sub_rank3 + obj_rank3

            total_loss += cur_loss.item() if hasattr(cur_loss, 'item') else float(cur_loss)
            # total_loss3 += cur_loss3.item() if hasattr(cur_loss3, 'item') else float(cur_loss3)
            batch_count += 1

    model.train()

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

    return s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3, avg_loss