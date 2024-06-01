## Backpropagation Algorithm & SGD & CNN Implementation

擔任深度學習助教時寫給同學的教學範例程式

```
├─Backpropagation Algorithm to Classify MNIST
├─CNN to Classify FashionMNIST
└─Simple Stochastic Gradient Descent Implementation
    ├─Lab1-1
    └─Lab1-2
```

## Backpropagation Algorithm to Classify MNIST
我的文章：[Deep Learning - Backpropagation Algorithm to Classify MNIST](https://lavinia0724.github.io/2024/05/13/Deep-Learning-Backpropagation-Algorithm-to-Classify-MNIST/)

- `backpropagation9.py`
	- 純用 Python 手刻的 Backpropagation Algorithm to Classify MNIST
- `backpropagation9_pytorch.py`
	- 透過 PyTorch 撰寫的 Backpropagation Algorithm to Classify MNIST
- `mnist_train9.csv`
	- 訓練資料集，共有 60,000 筆
- `mnist_test9.csv`
	- 測試資料集，共有 10,000 筆
- `mnist_ans9.csv`
	- 測試資料集的標準答案

## CNN to Classify FashionMNIST
我的文章：[Deep Learning - CNN to Classify FashionMNIST](https://lavinia0724.github.io/2024/05/24/Deep-Learning-CNN-to-Classify-FashionMNIST/)
- `CNN_main.py` : 透過 PyTorch 撰寫的 CNN to Classify FashionMNIST
- `best_model.pth` : 示範的 best model，準確率大約 93%

## Simple Stochastic Gradient Descent Implementation
我的文章：[Deep Learning - Simple Stochastic Gradient Descent Implementation](https://lavinia0724.github.io/2024/05/08/Deep-Learning-Simple-Stochastic-Gradient-Descent-Implementation/)

- Lab1-1
	- `Lab1_traindata1.txt` : 輸入資料
	- `output.png` : 輸出圖
	- `main.py` : 自行撰寫隨機梯度下降的線性回歸程式
- Lab1-2
	- `Lab1_traindata2.csv` : 輸入資料
	- `main.py` : 自行撰寫隨機梯度下降的線性回歸程式


## Reference
- 黃貞瑛老師的深度學習課程
- 許瀚丰、應名宥學長的助教輔導課程
- 吳建中、詹閎安同學的共同討論
