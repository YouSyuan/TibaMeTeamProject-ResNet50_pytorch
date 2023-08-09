import torch
from tqdm.auto import tqdm

def train_epoch(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)  # number of smples.
    num_batches = len(dataloader)  # batches per epoch.

    model.train()

    epoch_loss, epoch_correct = 0, 0
    for batch_i, (x,y) in enumerate(tqdm(dataloader)):
        x, y = x.to(device), y.to(device)  # move data to GPU.
        optimizer.zero_grad()  # zero the parameter gradients. 將梯度參數歸零

        # Compute prediction loss 計算預測損失
        pred = model(x)
        loss = loss_fn(pred, y)

        # Optimization by gradients. 梯度優化
        loss.backward()  # backpropagation to compute gradients. 反向傳播計算梯度
        optimizer.step()  # update model params. 更新模型參數

        # write to logs
        epoch_loss += loss.item()  # tensor -> python value
        epoch_correct += (pred.argmax(dim=1) == y).sum().item()
    
    # return avg loss of epoch, acc of epoch. 返回 epoch 的平均損失和準確率
    return epoch_loss/num_batches, epoch_correct/size


def val_epoch(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)  # number of samples.
    num_batches = len(dataloader)  # batches per epoch.

    model.eval()  # model to test mode.  將模型設定為「預測模式」
    
    epoch_loss, epoch_correct = 0, 0

    # No gradient for test data. 在預測階段，不需要進行梯度計算和參數更新
    with torch.no_grad():
        # 在該區塊內所計算出的 tesor 的 requires_grad 都會自動設置為 False
        for batch_i, (x,y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction loss 計算預測損失
            pred = model(x)
            loss = loss_fn(pred, y)

            # write to logs
            epoch_loss += loss.item()  # tensor -> python value
            epoch_correct += (pred.argmax(dim=1) == y).sum().item()

    # return avg loss of epoch, acc of epoch. 返回 epoch 的平均損失和準確率
    return epoch_loss/num_batches, epoch_correct/size