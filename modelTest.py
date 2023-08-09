import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from resnet50model import myResNet50
from dataset import build_dataloader
from class_name import CLASSES_NAME

# Pytorch
import torch



def predict(model_path, img_paths, batch_size=16, device="cpu"):
    img_size = 512
    classes_map = {name:i for i, name in enumerate(CLASSES_NAME)}
    class_number = len(CLASSES_NAME)

    # Load model weights
    model = myResNet50(class_number)
    model.load_state_dict(torch.load(model_path))
    # Move model to GPU
    model.to(device)
    # Set model to evaluation mode
    model.eval()

    # Build dataloader
    pre_loader = build_dataloader(img_paths, classes_map, 512, batch_size)


    # Predict
    pre_correct = 0
    pre_fail = 0
    y_true = torch.tensor([], dtype=torch.int64).to(device)
    y_pred_logits = torch.tensor([]).to(device)
    
    for i, (x,y) in enumerate(tqdm(pre_loader)):  # 遍歷驗證集中的每個批次數據
        x, y = x.to(device), y.to(device)
            
        # 預測時不需要計算梯度
        with torch.no_grad():
            pred = model(x)
        
        # 記錄所有預測值與正確答案
        y_true = torch.cat((y_true, y), dim=0)
        y_pred_logits = torch.cat((y_pred_logits, pred), dim=0)
    
    y_pred_cls = y_pred_logits.argmax(1) # get class idx with max prob   從預測結果中取出每個樣本最大概率所對應的類別索引
    y_pred_probs = torch.nn.Softmax(dim=1)(y_pred_logits)   # 將預測結果（logits）轉換為預測概率

    # Move tensor to CPU and convert to np.ndarray
    y_true = y_true.cpu().numpy()
    y_pred_cls = y_pred_cls.cpu().numpy()
    y_pred_probs = y_pred_probs.cpu().numpy()

    return y_true, y_pred_cls, y_pred_probs  # 真實標籤, 預測類別的索引, 該預測每個類別的概率







if __name__ == "__main__":
    import glob
    from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
    
    WEIGHT_PATH = "run/train1/best.pt"
    TEST_IMG_PATH = glob.glob("test/*")
    print(len(TEST_IMG_PATH))
    true, pre_cls, pre_probs = predict(WEIGHT_PATH, TEST_IMG_PATH, batch_size=16, device="cuda")

    fp_rate, tp_rate, threshold = roc_curve(true, pre_probs)

    torch.cuda.empty_cache()