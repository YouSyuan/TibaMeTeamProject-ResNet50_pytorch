import os, glob

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

import matplotlib.pyplot as plt
from PIL import Image



def create_folder(mode):
    os.makedirs("run", exist_ok=True)
    run_folders = [folder.split("\\")[-1] for folder in glob.glob("run/*") ]   
    n = 1    
    while True:
        folder_name = f"{mode}{n}"

        if folder_name not in run_folders:
            # 如果資料夾不存在，則直接建立資料夾
            print("Create:", folder_name)
            folder_name = f"run/{folder_name}"
            os.makedirs(folder_name)
            return folder_name
        else:
            # 如果資料夾已存在，則名稱後面加1
            n += 1

def save_chart(logs, logs_path):
    plt.figure(figsize=(12, 4))  # 創建一個新的畫布，設定大小為寬度12單位，高度4單位
    plt.subplot(1, 2, 1)  # 將畫布分成1行2列，並定位到第1個子圖
    plt.plot(logs['train_acc'])  # 繪製曲線
    plt.plot(logs['val_acc'])
    plt.legend(['train_acc', 'val_acc'])  # 加上圖例
    plt.xlabel('Epoch')  # 添加 x 軸標籤和 y 軸標籤
    plt.ylabel('Accuracy')  
    plt.title('Accuracy')  # 添加標題

    plt.subplot(1, 2, 2)
    plt.plot(logs['train_loss'])
    plt.plot(logs['val_loss'])
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss') 
    plt.title('Loss')

    # plt.show()
    plt.savefig(f'{logs_path}/loss.png')



def filling(img, size, save_path="", save=True, back_ground=(0,0,0)):
    """ 
    將圖片填充成正方形 
    調整成 size * size
    """

    # 取得原始圖片的尺寸
    width, height = img.size

    # 找出較大的一邊
    max_size = max(width, height)

    # 新建一個空白的正方形圖片
    square_img = Image.new('RGB', (max_size, max_size), back_ground)

    # 計算將原始圖片放入正方形中的位置
    paste_x = (max_size - width) // 2
    paste_y = (max_size - height) // 2

    # 將原始圖片貼上正方形中
    square_img.paste(img, (paste_x, paste_y))

    # 調整大小成為目標大小
    img = square_img.resize((size, size))

    # 儲存新圖片
    if save:
        img.save(save_path)
    
    return img



def cutting(img, prop=0.4, place=1, save=False, save_path=""):
    """ 
    裁切圖片 
    0.裁上方  1.裁中間  2.裁下面  其他:原圖
    """

    w, h = img.size
    h_resize = int(h * prop)
    if place == 0:
        img =  img.crop((0, 0, w, h_resize))  # (左上角座標x1,y1,右下角座標x2,y2)
    elif place == 1:
        y = int(h * ((1-prop) / 2))
        img =  img.crop((0, y, w, y+h_resize))
    elif place == 2:
        y = int(h * (1-prop))
        img =  img.crop((0, y, w, h))

    
    if save:
        img.save(save_path)
    
    return img


import seaborn as sns
def dataCollation(true, pre_cls, pre_probs, classes_name, weight_path, f1c=False, cfm=False, roc=False):
    save_path = "/".join(weight_path.split("/")[:-1])

    c = classification_report(true, pre_cls, digits=4, target_names=classes_name, zero_division=0)
    print(c)

    if f1c:
        # F1 scores  (還會報錯)
        F1_scores = f1_score(true, pre_cls, average='macro')  # 計算 F1 分數
        Precision_score = precision_score(true, pre_cls)  # 計算精確率
        Acc_score = accuracy_score(true, pre_cls)  # 計算準確率
        Recall_score = recall_score(true, pre_cls)  # 計算召回率
        print("F1 scores", F1_scores)
        print("precision_score: ", Precision_score)
        print("Acc score: ", Acc_score)
        print("recall_score: ", Recall_score)
        with open(f"{save_path}/F1_scores", "w") as file:
            file.write(f"F1 scores:       {F1_scores}\n")
            file.write(f"precision_score: {Precision_score}\n")
            file.write(f"Acc score:       {Acc_score}\n")
            file.write(f"recall_score:    {Recall_score}\n")


    if cfm:
        # 混淆矩陣
        cm = confusion_matrix(true, pre_cls)
        fig, ax = plt.subplots(figsize=(40, 30), dpi=100)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_name)
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        sns.set(font_scale=1.2)
        plt.ylabel("True Label", fontsize=50)
        plt.xlabel("Predict Label", fontsize=50)
        plt.xticks(fontsize=20, rotation=45)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{save_path}/ConfusionMatrix.png')
        # plt.show()


    if roc:
        # ROC
        fp_rate, tp_rate, threshold = roc_curve(true, pre_probs)
        # AUC score
        auc_score = auc(fp_rate, tp_rate)
        print(f'AUC: {auc_score:.4f}')
        # ROC curve
        plt.xlabel('False Positive Rate (FPR) 1-Specificity')
        plt.ylabel('True Positive Rate (TPR) Sensitivity')
        plt.plot(fp_rate, tp_rate, marker="^")
        plt.title('ROC curve')
        plt.show()

        # print FP TF Threshold
        for fp, tp, thresh in zip(fp_rate, tp_rate, threshold):
            print(f'fp: {fp:.3f} recall: {tp:.3f} threshold: {thresh:.3f}')

    
