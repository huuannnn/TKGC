import torch
from tqdm import tqdm
from dataset import TKGDataLoader


def _move_batch_to_device(batch_data, device):
    """Shared helper — dùng chung với Trainer và test.py."""
    def to_tensor(x, dtype=None):
        """Convert numpy array hoặc tensor sang device, giữ dtype nếu cần."""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if dtype is not None:
            x = x.to(dtype=dtype)
        return x.to(device)

    return [
        to_tensor(batch_data[0]),                  # quadruples — giữ nguyên dtype (long)
        batch_data[1],                              # s_history  (list, không phải tensor)
        batch_data[2],                              # o_history  (list, không phải tensor)
        to_tensor(batch_data[3], torch.float32),   # s_label
        to_tensor(batch_data[4], torch.float32),   # o_label
        to_tensor(batch_data[5], torch.float32),   # s_frequency
        to_tensor(batch_data[6], torch.float32),   # o_frequency
    ]


def execute_valid(batch_size, total_data_tensor, model,
                  data,
                  s_history, o_history,
                  s_label, o_label,
                  s_frequency, o_frequency):
    """
    Args:
        total_data_tensor: torch.Tensor đã ở trên device (convert + cache ở Trainer,
                           không convert lại mỗi lần gọi).
    """
    s_ranks2, o_ranks2, all_ranks2 = [], [], []
    s_ranks3, o_ranks3, all_ranks3 = [], [], []
    total_loss2 = 0.0
    total_loss3 = 0.0
    batch_count = 0

    device = total_data_tensor.device

    valid_loader = TKGDataLoader(
        data, s_history, o_history,
        s_label, o_label,
        s_frequency, o_frequency,
        batch_size
    )

    model.eval()
    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Validating", unit='batch')
        for batch_data in pbar:
            batch_data = _move_batch_to_device(batch_data, device)

            _, _, _, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, 'Valid', total_data_tensor)

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            all_ranks2 += sub_rank2 + obj_rank2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            all_ranks3 += sub_rank3 + obj_rank3

            total_loss2 += cur_loss2.item() if hasattr(cur_loss2, 'item') else float(cur_loss2)
            total_loss3 += cur_loss3.item() if hasattr(cur_loss3, 'item') else float(cur_loss3)
            batch_count += 1

    model.train()

    avg_loss = (total_loss2 + total_loss3) / 2.0 / batch_count if batch_count > 0 else 0.0

    return s_ranks2, o_ranks2, all_ranks2, \
           s_ranks3, o_ranks3, all_ranks3, \
           avg_loss