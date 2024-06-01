import numpy as np
import matplotlib.pyplot as plt

# 讀取資料
data = np.loadtxt(fname="Lab1_traindata2.csv",
                  dtype=np.float64,
                  delimiter=",",
                  encoding="utf-8-sig")     # (52 , 5)

# 資料切割
X = data[: , :-1]   # (52 , 4)
Y = data[: , -1:]   # (52 , 1)

# 初始化 w 和 b
# 這邊用的是 uniform 均勻分布
# 同學們可以試試看不同的初始化方法
w = np.random.uniform(low=-1. , high=1. , size=(4 , 1))  # (4 , 1)
b = np.random.uniform(low=-1. , high=1.)

# 設定超參數
max_epoch = 1000        # 最大世代數
learning_rate = 1e-6    # 學習率
tau = 1e-1              # 提前終止條件

for epoch in range(max_epoch):
    for i in range(len(X)):

        # 提取資料
        x = X[i].reshape(1 , 4)  # (1 , 4)
        y = Y[i]

        # 預測 ŷ = b + wx
        y_predict = np.matmul(x , w) + b   # (1 , 4) * (4 , 1)
        
        # 損失函數 = 0.5 * (y_predict - y) ** 2
        # 計算梯度
        w_gradient = (y_predict - y) * x.T
        b_gradient = y_predict - y

        # 依照梯度反方向更新 w 和 b （梯度下降）
        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

    # 計算 mse: 1/n Σ(y - ŷ)²
    Y_predict = np.matmul(X , w) + b  # (52 , 4) * (4 , 1)
    mse = np.mean((Y_predict - Y) ** 2)

    # 判斷是否提前終止
    if mse < tau:
        break


# 預測題目要求的兩筆資料
xa = np.array([[6.8 , 210 , 0.402 , 0.739]])
xb = np.array([[6.1 , 180 , 0.415 , 0.713]])

# ŷ = b + wx
ŷa = np.matmul(xa , w) + b  # (1 , 4) * (4 , 1)
ŷb = np.matmul(xb , w) + b  # (1 , 4) * (4 , 1)


# 題目輸出要求
print("-------------------------------------------")
print("深度學習 Lab1-2 輸出:")
print(f"w = {w}")
print(f"b = {b}")

print(f"當 x = {xa.reshape(-1)} , ŷ = {ŷa.item():.5f}")
print(f"當 x = {xb.reshape(-1)} , ŷ = {ŷb.item():.5f}")
print("------------------------------------------")
