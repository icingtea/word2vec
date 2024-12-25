import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int):
        
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        
        self.hidden_layer1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.layer_norm1 = nn.LayerNorm(embedding_dim // 2)
        
        self.hidden_layer2 = nn.Linear(embedding_dim // 2, embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        self.hidden_layer3 = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, context: torch.Tensor) -> torch.Tensor:

        embedding = self.embedding_layer(context).mean(dim=1)
        
        logits1 = F.gelu(self.layer_norm1(self.hidden_layer1(embedding)))
        logits2 = F.gelu(self.layer_norm2(self.hidden_layer2(logits1)))
        logits3 = self.hidden_layer3(logits2)
        
        return logits3