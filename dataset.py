from PIL import Image

import torch
import torchvision.transforms as transforms

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
        # cls_name = path.split("\\")[-1].split("_")[0]  # 95 個類別都要放
        # cls_name = path.split("_")[0]  # 單張
        cls_name = path.split("_")[3][3:7]
        cls_idx = self.cls_map[cls_name]
        cls_idx = torch.tensor(cls_idx, dtype=torch.int64)
        # cls_idx = cls_idx.unsqueeze(dim=0)

        return img, cls_idx  
     
    

# Build dataloder
def build_dataloader(data_paths, classes_map, img_size=512, batch_size=16, shuffle=False):
    # Preprocess Transform  預處理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Build dataset
    ds = MyDataset(data_paths, classes_map, transform)

    # Build dataloader
    dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=shuffle)

    return dl  





if __name__ == "__main__":
    import glob
    
    class_map = {'A01N': 0, 'A02W': 1, 'A02E': 2, 'A03W': 3, 'A03E': 4, 'A04N': 5, 'A04S': 6, 'A05S': 7, 'A05N': 8, 'A06W': 9, 'A06E': 10, 'A07E': 11, 'A07W': 12, 'A08S': 13, 'A08N': 14, 'A09W': 15, 'A09E': 16, 'A10W': 17, 'A10E': 18, 'A11E': 19, 'A11W': 20, 'A12W': 21, 'A12E': 22, 'A13W': 23, 'A13E': 24, 'A14W': 25, 'A14E': 26, 'A15W': 27, 'A15E': 28, 'A16N': 29, 'A17W': 30, 'A18W': 31, 'A19S': 32, 'A20S': 33, 'A20N': 34, 'A21E': 35, 'A21W': 36, 'A22N': 37, 'B01S': 38, 'B01N': 39, 'B02S': 40, 'B02N': 41, 'C01E': 42, 'C01W': 43, 'C02W': 44, 'C03W': 45, 'C04N': 46, 'C04S': 47, 'C05S': 48, 'C06E': 49, 'C06W': 50, 'C07S': 51, 'C07N': 52, 'C08E': 53, 'C08W': 54, 'C09W': 55, 'C09E': 56, 'C10E': 57, 
'C10W': 58, 'C11E': 59, 'C11W': 60, 'C12E': 61, 'C12W': 62, 'C13E': 63, 'C13W': 64, 'C14W': 65, 'C14E': 66, 'C15S': 67, 'C15N': 68, 'C16E': 69, 'C16W': 70, 'C17S': 71, 'C18E': 72, 'C19S': 73, 'D01N': 74, 'D01S': 75, 'D02E': 76, 'D03N': 77, 'D03S': 78, 'E01S': 79, 'E01N': 80, 'E02N': 81, 'E02S': 82, 'E03S': 83, 'E03N': 84, 'E04N': 85, 'E04S': 86, 'E05S': 87, 'E05N': 88, 'E06S': 89, 'E06N': 90, 'E07S': 91, 'E07N': 92, 'E08S': 93, 'E08N': 94}

    WEIGHT_PATH = "run/train1/best.pt"
    TEST_IMG_PATH = glob.glob("H:/TibaMe_TeamProject/projects/img_no_SS/test/*/*")[:20]
    print(len(TEST_IMG_PATH))
    test = build_dataloader(TEST_IMG_PATH, class_map)
    print(test)
    for i,j in test:
        print(i.shape, j)