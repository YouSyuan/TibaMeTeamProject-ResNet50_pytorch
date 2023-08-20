import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# pytorch
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

# cam
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    
# myfc
from resnet50model import myResNet50


def read_img(filepath, img_size):
    img = Image.open(filepath).resize((img_size, img_size))
    return img

def visualize(img, method, img_size, target_category=None):
    """

    如果 target_category 為 None，則得分最高的類別
    將用於批次中的每個圖像

    target_category 也可以是一個整數，或不同整數的列表
    對於批次中的每個圖像
    """

    #將圖像進行預處理與轉換到 torch 張量，並添加維度 dim
    input_tensor = transform(img).unsqueeze(0)  # (1, 3, img_size, img_size)
    print(input_tensor.shape)
    # 建構一個 CAM 物件，可以在其他的圖像讓面重複使用
    cam = method(model=model, target_layers=target_layer, use_cuda=True)
   
    # 調用 CAM 對象的 call 方法，將輸入圖像張量 input_tensor 和目標類別 target_category 傳入，生成灰度的 CAM 圖
    grayscale_cam = cam(input_tensor=input_tensor, 
                        targets=target_category,
                        # target_category=target_category,
                        )
    
    grayscale_cam = grayscale_cam[0, :]
    # 將 CAM 圖覆蓋在原始圖像上，得到在原始圖像上的 CAM 可視化結果
    cam_on_img = show_cam_on_image(np.array(img)/255., grayscale_cam, use_rgb=True)

    return grayscale_cam, cam_on_img



import glob
paths = glob.glob("H:/TibaMe_TeamProject/projects/img_data/data/test_SS/A12E/*")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512


# transform = ResNet50_Weights.DEFAULT.transforms()
# model = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 95)
model.load_state_dict(torch.load("run/train2_AdamW/best.pt"))
model.to(DEVICE)
# print(model) 
   

target_layer = [model.layer4[-1]]

for path in paths:
    img = read_img(path, IMG_SIZE)
    img = transform(img).unsqueeze(0).to(DEVICE)
    pred = model(img)

    # convert to probabilities of each classes
    pred = torch.nn.Softmax(dim=1)(pred)[0]

    print('Top 1 class: ', pred.argmax().item())
    values, ids =  torch.topk(pred, k=5)
    print('Top 5 class: ', ids.tolist())
    print('Top 5 value: ', values.tolist())

    img = read_img(path, IMG_SIZE)
    grayscale_cam, cam_on_img = visualize(img.copy(),
                                            GradCAM,
                                            img_size=IMG_SIZE,
                                            target_category=None)
    save_path = path.split("\\")[-1]

    
    # plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cam_on_img)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(grayscale_cam, cmap="jet")
    plt.axis("off")
    plt.savefig(save_path)
    # plt.show()