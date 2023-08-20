from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, cls_map, transform):
        self.img_paths = img_paths
        self.cls_map = cls_map
        self.transform = transform

    def __len__(self):
        # Number of samples.  樣本數量
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Read img.
        path = self.img_paths[idx] # get img path
        img = Image.open(path).convert('RGB')

        # Transform img to tensor (3, H, W).
        img = self.transform(img)

        # Read class index.
        # cls_name = path.split("\\")[-1].split("_")[0] 
        # cls_name = path.split("_")[0]  
        cls_name = path.split("/")[-2]
        cls_idx = self.cls_map[cls_name]
        cls_idx = torch.tensor(cls_idx, dtype=torch.int64)
        # cls_idx = cls_idx.unsqueeze(dim=0)

        return img, cls_idx  
     
    

# Build dataloder
def build_dataloader(img_paths, classes_map, img_size=512, batch_size=16, shuffle=False):
    # Preprocess Transform  預處理
    # transform = ResNet50_Weights.DEFAULT.transforms()
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Build dataset
    ds = MyDataset(img_paths, classes_map, transform)

    # Build dataloader
    dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=shuffle)

    return dl  




