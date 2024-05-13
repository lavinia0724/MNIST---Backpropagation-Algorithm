import torch
from torch.utils import data as data_
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 設定要跑的世代數量
epochs = 100

# 讀檔，透過 pandas 的 df.read_csv 來讀 csv 檔
df = pd.read_csv('mnist_train9.csv')

# df.iloc，dataframe integer location 用整數位置做為基準去讀資料，
# [:, 1:]，row 全部資料都讀，cloumn 是從第 1 格開始讀到最後
# 變數型態是 float32
# .values/255 是做資料的標準化，有利於神經網路的訓練，以避免梯度消失或爆炸的問題
# 最後轉換成 torch 的 tensor 型態
train_data_x = torch.tensor(df.iloc[:, 1:].values/255, dtype=torch.float32)
# [:, 0] y 是讀 label，只要讀第 0 行就好
# pd.get_dummies 是利用 pandas 做 one-hot encoding 的方式
train_data_y = torch.tensor(pd.get_dummies(df.iloc[:, 0]).values, dtype=torch.float32)

# 從資料集中取 80% 當 training dataset、20% 當 validation dataset
train_data_amount = (len(train_data_x) // 10) * 8

# validation 取剛才取出的 dataset 中從 80% 開始到最後的所有資料
validation_data_x = train_data_x[train_data_amount:]
validation_data_y = train_data_y[train_data_amount:]

# train 取從頭開始到 80% 的資料
train_data_x = train_data_x[:train_data_amount]
train_data_y = train_data_y[:train_data_amount]


# 定義神經網絡模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        # super 的用法就是將函數的調用委託給父類別，而通常那個父類別就是pytorch內建的函數 nn.Module 所以我們需要用 __init__() 來初始化(initialize)整個函數
        super(NeuralNetwork, self).__init__()

        # 建立 3 層神經網路，每層的輸入與輸出 channel 如下
        self.fc1 = nn.Linear(784, 30) # 784 row(in) x 30 cloumn(out)
        self.fc2 = nn.Linear(30, 28) # 30 row x 28 cloumn
        self.fc3 = nn.Linear(28, 9) # 28 row x 9 clumn

        # 每層神經網路的權重都透過常態分佈來設計，標準差為 0.1，平均為 0.0
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.1)

    def forward(self, x):
        x = x.view(1, 784)        # 吃進來的一筆資料總共 1 row 784 column
        x = torch.sigmoid(self.fc1(x))   # 對第一層神經網路的 x 輸出做 sigmoid
        x = torch.sigmoid(self.fc2(x))   # 對第二層神經網路的 x 輸出做 sigmoid
        x = self.fc3(x)
        return torch.softmax(x, dim=1)   # 對第三層神經網路的 x 輸出做 softmax

# 設計 Early Stop 條件
class Early_Stop_checker:
    def __init__(self, patience=1):
        self.patience = patience  # patience 為能夠接受 validation loss 減少變好的世代數
        self.counter = 0          # counter 用來計算目前已經幾代 validation loss 沒有變好
        self.min_validation_loss = float('inf') # 紀錄所有世代中最小的 validation loss

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss: # 如果當前的 validation loss 比過去最好的 validation loss 要小，則更新 min_validation_loss
            self.min_validation_loss = validation_loss
            self.counter = 0                           # 更新後 counter 歸零
        elif validation_loss > self.min_validation_loss: # 如果當前的 validation loss 比過去最好的 validation loss 要不好
            self.counter += 1                          # counter + 1
            if self.counter >= self.patience:          # 如果 counter 已經超過 patience，就 early stop
                return True
        return False

# 初始化模型、損失函數和優化器
model = NeuralNetwork()
# 使用交叉熵損失函數
criterion = nn.CrossEntropyLoss()
# 使用隨機梯度下降演算法，learning rate 設 0.01
optimizer = optim.SGD(model.parameters(), lr = 0.01)


early_stop_checker = Early_Stop_checker(patience=3)

# 訓練模型
for epoch in range(epochs):
    # 使用 Stochastic 梯度下降，要一筆一筆資料跑，batch 或 mini-batch 才是一口氣跑多筆資料
    for i in range(len(train_data_x)):
        # 訓練模型要設定成 model.train()
        model.train()

        # 清除所有梯度
        optimizer.zero_grad()
        # 將 train_data_x 一筆一筆放進模型訓練，將該筆資料的預測值存進 predict
        # torch.unsqueeze 對 x 做擴維，原本的 torch.tensor(x) = [1, 2, 3, 4]，擴維後變成 [[1, 2, 3, 4]]
        predict = model(torch.unsqueeze(train_data_x[i], 0))

        # 計算 loss，把預測的 predict (也就是 ŷ)，和 y 去做 cross entropy 計算
        # 將 y 擴維後透過 torch.argmax 返回指定維度
        loss = criterion(predict, torch.unsqueeze(train_data_y[i], 0).argmax(dim=1))

        # 做 Backpropagation
        loss.backward() # 透過反向傳播獲得每個參數的梯度值
        optimizer.step() # 透過梯度下降執行參數更新

    # 計算這次世代更新完後的 loss 和 accuracy
    # torch.no_grad() 顧名思義就是 no gradient
    with torch.no_grad():
        # 評估模型要設定成 model.eval()
        model.eval()

        # ----------------------------- training ----------------------------------------------- #
        train_loss = 0

        for i in range(len(train_data_x)):
            # 一樣把 train data x 丟給模型預測，只是這次不用再計算梯度和更新
            train_predict = model(torch.unsqueeze(train_data_x[i], 0))
            # 加總全部的 loss (等等最後會取平均)
            train_loss += criterion(train_predict, torch.unsqueeze(train_data_y[i], 0).argmax(dim=1))

        # 將所有資料的 loss (剛剛加總了)取平均
        train_loss = train_loss/len(train_data_x)

        # ----------------------------- validation ----------------------------------------------- #
        validation_loss = 0

        # 驗證其實做的是一樣的事情，只是在模型訓練的時候，並沒有學過這些資料
        for i in range(len(validation_data_x)):
          # 把要驗證的資料都進已經訓練好的模型預測
          validation_predict = model(torch.unsqueeze(validation_data_x[i], 0))
          # 計算驗證 loss
          validation_loss += criterion(validation_predict, torch.unsqueeze(validation_data_y[i], 0).argmax(dim=1))

        # 計算驗證 loss
        validation_loss = validation_loss/len(validation_data_x)

        # Early Stopping，提前終止訓練，為了避免 overfitting
        if early_stop_checker.early_stop(validation_loss):
            print(f"Early stopping at epoch: {epoch} ")
            break

        # 輸出當前世代、當前的 train loss、validation loss
        print("----------------------")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print("----------------------")

train_correct = 0
validation_correct = 0

# 計算 train accuracy 和 validation accuracy
for i in range(len(train_data_x)):
    train_predict = model(torch.unsqueeze(train_data_x[i], 0))
    # torch.max(train_predict, 1) 回傳的是在 train_predict 中有最大值的 index，也就會是該筆資料的 label
    # (train_predict 回傳的是此筆資料是 1, 2, 3, 4, 5, 6, 7, 8, 9 的機率，所以取最大機率的那格，就是預測此筆資料為那個數字)
    _, predicted_label = torch.max(train_predict, 1)

    # 如果預測成功了，預測成功的筆數就 + 1
    if(predicted_label == torch.argmax(train_data_y[i])):
        train_correct += 1

for i in range(len(validation_data_x)):
    validation_predict = model(torch.unsqueeze(validation_data_x[i], 0))

    # 確認驗證資料是否預測正確
    _, predicted_label = torch.max(validation_predict, 1)

    # 如果驗證預測正確，驗證預測正確筆數 + 1
    if(predicted_label == torch.argmax(validation_data_y[i])):
        validation_correct += 1

# 準確率為所有的資料中，預測成功的筆數
train_accuracy = train_correct / len(train_data_x)
validation_accuracy = validation_correct / len(validation_data_x)

# 輸出訓練結束、最後在哪個世代結束、最終的 train loss、train accuracy、validation loss、validation accuracy
print("----------------------")
print('Finished Training')
print(f"Epoch result {epoch}")
print(f"Train Loss: {train_loss:.4f}")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Loss: {validation_loss:.4f}")
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
print("----------------------")

# ---------------------- Testing -----------------------------------------------#
# 讀 test 資料
df = pd.read_csv('mnist_test9.csv')

# 同樣，讀所有 test 資料，但因為 test 資料不會有 label，所以 [:, :] 所有 row 和 column 都讀
test_data_x = torch.tensor(df.iloc[:, :].values/255, dtype=torch.float32)

# 因為最後要把 testing 每筆資料的預測結果輸出成 csv 檔，所以要記錄結果
test_predict_result = []
classification = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# testing 也是一種評估模型模式，一樣不用計算梯度
with torch.no_grad():
    model.eval()
    for i in range(len(test_data_x)):
      # 把 test 資料都進模型預測
      test_predict = model(torch.unsqueeze(test_data_x[i], 0))
      # 得到該筆資料的 label
      _, predicted_label = torch.max(test_predict, 1)

      # 把預測結果的 label 記錄起來
      test_predict_result.append(classification[predicted_label.numpy().item()])

# 把 test_predict_result 轉成 pandas 的 dataframe，並且給他欄位名稱叫做 Label
test_predict_result = pd.DataFrame({'Label': test_predict_result})
# 把 test_predict_result 輸出 csv 檔
test_predict_result.to_csv('test_predict_result.csv', index=False)

# 保存訓練好的模型，路徑為跟此程式相同路徑
PATH = './mnist_nn.pth'
torch.save(model.state_dict(), PATH)
