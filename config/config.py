import torch

class Config:
    
    CONTEXT_SIZE: int = 4
    EMBEDDING_DIM: int = 256
    LEARNING_RATE: float = 10e-4
    MIN_FREQ: int = 100
    NUM_EPOCHS: int = 15
    TRAIN_SPLIT: float = 0.8
    BATCH_SIZE: int = 8
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    GENERATOR: torch.Generator = torch.Generator(device = DEVICE)