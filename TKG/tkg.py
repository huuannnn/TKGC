import torch
import os
from .quadruple import Quadruple
from itertools import groupby
from typing import List
from torch_geometric.data import Data


class ReadableTKG:
    def __init__(self, entity2id_path=None, relation2id_path=None):
        self.entity2id = self.load_vocab(entity2id_path)
        self.relation2id = self.load_vocab(relation2id_path)
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

    @staticmethod
    def load_vocab(path):
        if path is None or not os.path.exists(path):
            return {}
        vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                try:
                    vocab[parts[0]] = int(parts[1])
                except ValueError:
                    print(f"[Warning] Skip malformed line {line_idx}: {line.strip()}")
        return vocab

    def add_inverse_relations(self, num_rels: int):
        for rel, idx in list(self.relation2id.items()):
            self.id2relation[idx + num_rels] = f"Inverse_{rel}"

    def quadruple_to_string(self, quadruple):
        if isinstance(quadruple, torch.Tensor):
            quadruple = Quadruple.from_tensor(quadruple)

        src = self.id2entity.get(quadruple.src, str(quadruple.src))
        rel = self.id2relation.get(quadruple.rel, str(quadruple.rel))
        dst = self.id2entity.get(quadruple.dst, str(quadruple.dst))

        return f"({src}) --[{rel}]--> ({dst}) @ t={quadruple.tim}"

    def snapshot_to_string(self, snapshot, max_lines=None):
        lines = [self.quadruple_to_string(row) for row in snapshot]
        if max_lines is not None:
            lines = lines[:max_lines]

        return "\n".join(lines)
      
class TemporalKnowledgeGraph:
    def __init__(self, quadruples: List, num_nodes: int, num_rels: int, add_inverse: bool = True):
        self.num_nodes = num_nodes
        self.raw_num_rels = num_rels
        self.add_inverse = add_inverse

        if add_inverse:
            quadruples = quadruples + [q.inverse(num_rels) for q in quadruples]
            self.num_rels = num_rels * 2
        else:
            self.num_rels = num_rels

        quadruples = sorted(quadruples, key=lambda x: x.tim)
        self.snapshots, self.snapshot_times = self.build_snapshots(quadruples)
        self.quadruples = torch.stack([q.to_tensor() for q in quadruples], dim=0)

    def build_snapshots(self, quadruples):
        snapshots = []
        snapshot_times = []

        for tim, group in groupby(quadruples, key=lambda x: x.tim):
            group = list(group)
            snapshot_times.append(tim)

            src = torch.tensor([q.src for q in group], dtype=torch.long)
            rel = torch.tensor([q.rel for q in group], dtype=torch.long)
            dst = torch.tensor([q.dst for q in group], dtype=torch.long)

            edge_index = torch.stack([src, dst], dim=0)
            snapshot = Data(edge_index=edge_index, edge_attr=rel, num_nodes=self.num_nodes)
            snapshots.append(snapshot)

        return snapshots, snapshot_times

    def get_quadruples_at(self, idx: int) -> torch.Tensor:
        snapshot = self.snapshots[idx]
        tim = self.snapshot_times[idx]

        src = snapshot.edge_index[0]
        dst = snapshot.edge_index[1]
        rel = snapshot.edge_attr
        tim_col = torch.full_like(src, tim)

        return torch.stack([src, rel, dst, tim_col], dim=1)  # [E, 4]

    @property
    def num_snapshots(self):
        return len(self.snapshots)

    def get_snapshot(self, idx: int):
        return self.snapshots[idx]

    def get_history(self, idx: int, history_len: int, dilate_len: int):
        history_indices = [idx - i * dilate_len for i in range(history_len, 0, -1)]
        return [self.snapshots[i] for i in history_indices]

    def stat(self, name="Graph"):
        num_edges = [snapshot.edge_index.size(1) for snapshot in self.snapshots]
        return (
            f"[{name}] "
            f"Nodes: {self.num_nodes}, "
            f"Relations: {self.num_rels}, "
            f"Snapshots: {self.num_snapshots}, "
            f"Max edges: {max(num_edges)}, "
            f"Min edges: {min(num_edges)}"
        )

    def __str__(self):
        return self.stat()

    def __repr__(self):
        return self.stat()