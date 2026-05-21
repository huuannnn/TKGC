import torch

def _move_batch_to_device(batch_data, device):
    """Shared helper — dùng chung với Trainer và valid.py."""
    def to_tensor(x, dtype=None):
        """Convert numpy array hoặc tensor sang device, giữ dtype nếu cần."""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if dtype is not None:
            x = x.to(dtype=dtype)
        return x.to(device)

    return [
        to_tensor(batch_data[0]),           # quadruples — giữ nguyên dtype (long)
        batch_data[1],                       # s_history  (list, không phải tensor)
        batch_data[2],                       # o_history  (list, không phải tensor)
        to_tensor(batch_data[3], torch.float32),  # s_label
        to_tensor(batch_data[4], torch.float32),  # o_label
        to_tensor(batch_data[5], torch.float32),  # s_frequency
        to_tensor(batch_data[6], torch.float32),  # o_frequency
    ]