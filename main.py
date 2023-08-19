import datetime, glob
from modelTrain import train
from sklearn.model_selection import train_test_split

import torch

from modelTrain import train
from modelTest import predict
from otherTools import dataCollation
from class_name import CLASSES_NAME

start = datetime.datetime.now()
# Parameters (train / predict)
# MODE = "train" 
MODE = "predict"

TRAIN_IMG_PATH = "H:/TibaMe_TeamProject/projects/img_data/data/train/*/*"
VAL_IMG_PATH = "H:/TibaMe_TeamProject/projects/img_data/data/val/*/*"
PRE_IMG_PATH = "H:/TibaMe_TeamProject/projects/img_data/data/test/A05S/*"
WEIGHT_PATH = "best.pt"

IMG_SIZE = 512
TRAIN_BATCH_SIZE = 80
PREDICT_BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 10
CONF = 70.00

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if MODE == "train":
    print(MODE)
    print("Device:", DEVICE)
    # Read image paths
    # img_paths = glob.glob(TRAIN_IMG_PATH)
    train_paths = glob.glob(TRAIN_IMG_PATH)
    val_paths = glob.glob(VAL_IMG_PATH)
    img_number = len(train_paths) + len(val_paths)
    print("Total Class:", len(CLASSES_NAME))
    print("Total Img:", img_number)
    

    # split dataset
    # train_paths, val_paths = train_test_split(img_paths, test_size=0.1)
    print("train:", len(train_paths))
    print("val:", len(val_paths))

    # Training
    T = train(train_paths, val_paths, CLASSES_NAME, IMG_SIZE, TRAIN_BATCH_SIZE, EPOCHS, PATIENCE, DEVICE)
    print(T)

elif MODE == "predict":
    print(MODE)
    img_paths = glob.glob(PRE_IMG_PATH)
    if len(img_paths) <= 0:
        print("找不到圖片")
    else:
        true, pre_cls, pre_probs = predict(WEIGHT_PATH, img_paths, batch_size=PREDICT_BATCH_SIZE, device=DEVICE)
        
        if len(img_paths) > 1:
            dataCollation(true, pre_cls, pre_probs, CLASSES_NAME, WEIGHT_PATH, False, True, False)
        else:
            pre_probs = round((pre_probs[0][pre_cls[0]] * 100), 2)
            if pre_probs >= CONF and true[0] == pre_cls[0]:
                print(f"正確OO!!  正確答案：{CLASSES_NAME[true[0]]}  預測答案:{CLASSES_NAME[pre_cls[0]]}  信心值:{pre_probs}%")
            else:
                print(f"錯誤XX!!  正確答案：{CLASSES_NAME[true[0]]}  預測答案:{CLASSES_NAME[pre_cls[0]]}  信心值:{pre_probs}%")

else:
    print("Enter Error!")

torch.cuda.empty_cache()
end = datetime.datetime.now()
print("花費時間：", end-start)
