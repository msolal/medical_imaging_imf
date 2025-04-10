import torch

def collate_fn(batch):
    return (
        torch.stack(list(x[0].image.tensor for x in batch)),
        torch.stack(list(x[1].image.tensor for x in batch))
    )