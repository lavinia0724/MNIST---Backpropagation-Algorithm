import numpy as np
import matplotlib.pyplot as plt

# 讀取資料
data = np.loadtxt(fname="Lab1_traindata1.txt",
          dtype=np.float64,
          delimiter=None,  # 因為默認就是用空白隔開
          encoding="utf-8-sig")     # (100 , 2)

# 資料切割
X = data[: , 0]   # (100 , 1)
Y = data[: , 1]   # (100 , 1)

# 初始化 w 和 b
# 這邊用的是 normal 常態分布
# 同學們可以試試看不同的初始化方法
w = np.random.normal(loc=0 , scale=1)  
b = np.random.normal(loc=0 , scale=1)

# 設定超參數
max_epoch = 1000       # 最大世代數
learning_rate = 1e-6   # 學習率
tau = 1e-1             # 提前終止條件

for epoch in range(max_epoch):
    for i in range(len(X)):

        # 提取資料
        x = np.array([X[i]]) # 因為 w 是 np.array，x 也要是相同型態
        y = Y[i]

        # 預測 ŷ = b + wx
        y_predict = x * w + b  # 因為 x 是一維所以才能這樣乘，否則要用 np.matmul
           
        # 損失函數 = 0.5 * (y_predict - y) ** 2
        # 計算梯度
        w_gradient = (y_predict - y) * x.T # 雖然 x 是 (1 , 1) 但正確邏輯觀念是 x.T
        b_gradient = y_predict - y

        # 依照梯度反方向更新 w 和 b （梯度下降）
        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

    # 計算 mse: 1/n Σ(y - ŷ)²
    Y_predict = X * w + b  # (100 , 1) * (1 , 1)
    mse = np.mean((Y_predict - Y) ** 2)

    # 判斷是否提前終止
    if mse < tau:
        break

# 預測題目要求的兩筆資料
xa = 28.2
xb = 135.0

# ŷ = b + wx
ŷa = xa * w + b  
ŷb = xb * w + b  

# 題目輸出要求
print("-------------------------------------------")
print("深度學習 Lab1-1 輸出:")
print(f"w = {w.item():.5f}")
print(f"b = {b.item():.5f}")

print(f"當 x = {xa} , ŷ = {ŷa.item():.5f}")
print(f"當 x = {xb} , ŷ = {ŷb.item():.5f}")
print("------------------------------------------")

# plot 畫圖
plt.plot(xa, ŷa, 'o', color="salmon") # 畫點 xa 和 xb，圖形為 o
plt.plot(xb, ŷb, 'o', color="plum")

# 畫線性方程式，假設 100 個 x，套入 w 和 b 計算 ŷ
x_values = np.linspace(0, 150, 100) # 從 0 到 150 中間取 100 個點
y_values = x_values * w + b # 算出每個 x 的 ŷ
plt.plot(x_values, y_values, color="turquoise") # 畫出線性方程式

# 標題
plt.title("Deep Learning Lab1-1")

# x y 座標名稱
plt.xlabel("x value")
plt.ylabel("y value")

# 右下角標示顏色線的名稱
plt.legend([f"(xa, ŷa) = ({xa}, {ŷa.item():.1f})", f"(xb, ŷb) = ({xb}, {ŷb.item():.1f})", "ŷ = b + wx"], loc = "lower right")
# plt.show() 用來自己檢視圖片用

# 輸出成 output 檔案
plt.savefig('output.png')