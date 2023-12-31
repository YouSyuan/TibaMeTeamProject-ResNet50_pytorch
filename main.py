import datetime, glob
import numpy as np
from modelTrain import train
from sklearn.model_selection import train_test_split

# pytorch
import torch

# my fc
from modelTrain import train
from modelTest import predict
from otherTools import dataCollation
from class_name import CLASSES_NAME


# Parameters (train / predict)
# MODE = "train" 
MODE = "val"
# MODE = "predict"


TRAIN_IMG_PATH = "H:/TibaMe_TeamProject/projects/img_data/data/train/*/*"
VAL_IMG_PATH = "H:/TibaMe_TeamProject/projects/img_data/data/val/*/*"
PRE_IMG_PATH = "H:/TibaMe_TeamProject/projects/img_data/data/test/*/*"
WEIGHT_PATH = "run/train2_AdamW/best.pt"
IMG_SIZE = 512
TRAIN_BATCH_SIZE = 80
PREDICT_BATCH_SIZE = 16
EPOCHS = 70
PATIENCE = 10
CONF = 80.00

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

if MODE == "train":
    print(MODE)
    print("Device:", DEVICE)
    start = datetime.datetime.now()
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
    end = datetime.datetime.now()
    print(T)
    print("花費時間：", end-start)

elif MODE == "predict":
    print(MODE)
    img_paths = glob.glob(PRE_IMG_PATH)
    if len(img_paths) <= 0:
        print("找不到圖片")
    else:
        start = datetime.datetime.now()
        true, pre_cls, pre_probs = predict(WEIGHT_PATH, img_paths, batch_size=PREDICT_BATCH_SIZE, device=DEVICE)
        end = datetime.datetime.now()
        if len(img_paths) > 1:
            dataCollation(true, pre_cls, pre_probs, CLASSES_NAME, WEIGHT_PATH, False, True, False)
        else:
            pre_probs = round((pre_probs[0][pre_cls[0]] * 100), 2)
            if pre_probs >= CONF and true[0] == pre_cls[0]:
                print(f"正確OO!!  正確答案：{CLASSES_NAME[true[0]]}  預測答案:{CLASSES_NAME[pre_cls[0]]}  信心值:{pre_probs}%")
            elif pre_probs < CONF:
                print(f"信心值過低，無法辨識!!  正確答案：{CLASSES_NAME[true[0]]}  預測答案:{CLASSES_NAME[pre_cls[0]]}  信心值:{pre_probs}%")
            else:
                print(f"錯誤XX!!  正確答案：{CLASSES_NAME[true[0]]}  預測答案:{CLASSES_NAME[pre_cls[0]]}  信心值:{pre_probs}%")
    print("花費時間：", end-start)
elif MODE == "val":
    print(MODE)
    imgs = glob.glob(PRE_IMG_PATH)
    img_paths = ["/".join(img.split("\\")) for img in imgs]
    print(img_paths[0])
    samples = len(img_paths)
    print(samples)
    print(img_paths[5].split("/")[-2])
    for img in img_paths:
        print("img:", img)
        true, pre_cls, pre_probs = predict(WEIGHT_PATH, img, batch_size=PREDICT_BATCH_SIZE, device=DEVICE)
        pre_probs = round((pre_probs[0][pre_cls[0]] * 100), 2)

        correct = 0
        err = 0
        none = 0
        if pre_probs >= CONF and true[0] == pre_cls[0]:
            correct += 1
        elif pre_probs < CONF:
            none += 1
        else:
            err += 1
    print("測試數量：", samples)
    print("正確數量：", true)
    print("總錯誤數量：", err+none)
    print("辨識錯誤數量：", err)
    print("無法辨識數量：", none)
    print("準確率：", true/samples)


else:
    print("Enter Error!")

torch.cuda.empty_cache()
