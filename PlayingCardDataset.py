from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)  # This will assume that the sub folders of the
        # directory have the class name for the image, and it will handle creating the labels for us, this can be done
        # manually, but this will make it more simple.

    def __len__(self):  # The DataLoader will need to know how many examples we have in the Dataset when created.
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]  # This will give the item and the class

    @property
    def classes(self):
        return self.data.classes
