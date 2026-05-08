import torch
import torch.nn as nn

from cfln.utils import to_real  # noqa: F401 (re-exported for convenience)


class ComplexEmbedding(nn.Module):
    def __init__(self, vocab_size, d_c):
        super().__init__()
        self.embed_real = nn.Embedding(vocab_size, d_c)
        self.embed_imag = nn.Embedding(vocab_size, d_c)
        nn.init.normal_(self.embed_real.weight, std=0.02)
        nn.init.normal_(self.embed_imag.weight, std=0.02)

    def forward(self, token_ids):
        return torch.complex(self.embed_real(token_ids), self.embed_imag(token_ids))
