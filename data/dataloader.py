from torch.utils.data import Dataset


class ObjectsDataset(Dataset):
    """
    Creates dataloader for object-centric data.

    Args:
        X: numpy array of observations
        Z: numpy array of ground-truth latents
        transform: torchvision transformation for data

    Returns:
        inferred batch of observations and ground-truth latents
    """

    def __init__(self, X, Z, transform):
        self.obs = X
        self.factors = Z
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        x = self.obs[idx]
        if self.transform != None:
            x = self.transform(x)
        factors = self.factors[idx]
        return x, factors
