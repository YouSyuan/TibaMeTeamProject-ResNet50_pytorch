import numpy as np
from tqdm.auto import tqdm

# pytorch
import torch
from torch import nn
from torchsummary import summary

# My code
from dataset import build_dataloader
from resnet50model import myResNet50
from epoch import train_epoch, val_epoch
from otherTools import *


def train(train_paths, val_paths, classes_name, img_size=512, batch_size=16, epochs=50, patience=10, device="cpu"):
    classes_map = {name:i for i, name in enumerate(classes_name)}
    class_number = len(classes_name)
    # Build model
    model = myResNet50(class_number)

    # Build dataloader
    train_loader = build_dataloader(train_paths, classes_map, img_size, batch_size, shuffle=True)
    val_loader = build_dataloader(val_paths, classes_map, img_size, batch_size)

    # Early stopping
    counter = 0
    best_loss = np.inf

    # Training
    loss_fn = nn.CrossEntropyLoss()  # 損失函數
    optimizer = torch.optim.Adam(model.parameters())  # 優化器
    # optimizer = torch.optim.AdamW(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model = model.to(device)

    logs = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    save_path = create_folder("train")
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_epoch(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_acc = val_epoch(val_loader, model, loss_fn, device)

        print(f"EPOCH: {epoch:05d}       train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f}        val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}")

        logs["train_loss"].append(train_loss)
        logs['train_acc'].append(train_acc)
        logs['val_loss'].append(val_loss)
        logs['val_acc'].append(val_acc)

        

        # On epoch end
        torch.save(model.state_dict(), f"{save_path}/last.pt")
        # Check improvement 檢查模型參數是否更好
        if val_loss < best_loss:
            counter = 0
            best_loss = val_loss
            torch.save(model.state_dict(), f"{save_path}/best.pt")
        else:
            counter += 1
        
        if counter >= patience:
            print("EarlyStop!!!")
            break        

    # 繪製折線圖 
    save_chart(logs, save_path)

    # Save model
    torch.save(model, f"{save_path}/model.pth")


    return "===== End of Training! ====="

