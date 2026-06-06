import torch
from dataclasses import dataclass

@dataclass
class Quadruple:
    src: int
    rel: int
    dst: int
    tim: int

    def inverse(self, num_rels: int):
        return Quadruple(src=self.dst, rel=self.rel + num_rels, dst=self.src, tim=self.tim)

    def to_tensor(self):
        return torch.tensor([self.src, self.rel, self.dst, self.tim], dtype=torch.long)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(src=int(tensor[0]), rel=int(tensor[1]), dst=int(tensor[2]), tim=int(tensor[3]))

    def as_tuple(self):
        return (self.src, self.rel, self.dst, self.tim)