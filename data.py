import glob
import cv2
from numpy import asarray, moveaxis

from torch.utils.data import Dataset, DataLoader


class VOCDataset(Dataset):
    def __init__(self, subset_indexes, input_folder, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.files = glob.glob(input_folder + "*")[subset_indexes]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lr = cv2.resize(img, (72, 72))
        mr = cv2.resize(img, (144, 144))
        hr = cv2.resize(img, (288, 288))
        return lr, mr, hr, moveaxis(lr, 2, 0), moveaxis(mr, 2, 0), moveaxis(hr, 2, 0)