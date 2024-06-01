import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid activation function
def Sigmoid(n):
    return 1 / (1 + np.exp(-n))
     
# One-Hot Encoding
def One_Hot_Encoding(y):
    return pd.get_dummies(y).to_numpy()

# Softmax: e^zi / Σ e^zj 
def Softmax(aL):
    # size: aL = 9 * 1
    for col in range(1):
        total = 0.0
        for row in range(len(aL)):
            total += np.exp(aL[row][col])
        for row in range(len(aL)):
            aL[row][col] = np.exp(aL[row][col]) / total
    return aL


# Load the training data
df = pd.read_csv('mnist_train9.csv')

# row size: 40500, col size: 785
rowx, colx = df.shape

# take 80% data to training, 20% data to verify 
training_data_amount = (rowx // 10) * 8

# row size: 40500, col size: 784
# avoid Sigmoid too large: [all data] / 1000
training_data_x = df.to_numpy()[ : training_data_amount, 1:] / 1000
verify_data_x = df.to_numpy()[training_data_amount : , 1:] / 1000

# row size: 9, col size: 1
training_data_y = df.to_numpy()[ : training_data_amount, 0]
verify_data_y = df.to_numpy()[training_data_amount : , 0]

# one hot encoding (y): 
training_data_y = One_Hot_Encoding(training_data_y)
verify_data_y = One_Hot_Encoding(verify_data_y)


# Parameter/Hyperparameter
epoch = 40
learning_rate = 0.01

# use normal distribution random number to init w
w1 = np.random.normal(loc = 0, scale = 0.1, size = (30, 784))
w2 = np.random.normal(loc = 0, scale = 0.1, size = (28, 30))
w3 = np.random.normal(loc = 0, scale = 0.1, size = (9, 28))

# init b
b1 = np.random.rand(30, 1)
b2 = np.random.rand(28, 1)
b3 = np.random.rand(9, 1)

# output information
output_epoch = []
output_training_correct_rate = []
output_verify_correct_rate = []


# training model ===================================

epoch_result = 0
for i in range (epoch):

    for j in range(len(training_data_x)):

        # size: (1, 784)
        x_tmp = []
        x_tmp.append(training_data_x[j])
        x_tmp = np.array(x_tmp)

        # size: (1, 9)
        y_tmp = []
        y_tmp.append(training_data_y[j])
        y_tmp = np.array(y_tmp)

        # feedforward -----------------------------------

        # size: w1 = 30 * 784, x_tmp.transpose() = 784 * 1, b1 = 30 * 1
        # size: a1 = 30 * 1
        n1 = w1 @ x_tmp.transpose() + b1
        a1 = Sigmoid(n1)

        # size: w2 = 28 * 30, a1 = 30 * 1, b2 = 28 * 1
        # size: a2 = 28 * 1
        n2 = w2 @ a1 + b2
        a2 = Sigmoid(n2)

        # size: w3 = 9 * 28, a2 = 28 * 1, b3 = 9 * 1
        # size: aL = 9 * 1
        n3 = w3 @ a2 + b3
        aL = Softmax(n3)

        # backward ------------------------------------

        # size: aL = 9 * 1, y_tmp = 1 * 9
        # size: delta_L = 9 * 1
        delta_L = (aL - y_tmp.transpose())

        # size: w3 = 9 * 26, delta_L = 9 * 1, a2 = 26 * 1
        # size: delta_2 = 26 * 1
        delta_2 = (w3.transpose() @ delta_L) * (a2 * (1 - a2))

        # size: w2 = 26 * 20, delta_2 = 26 * 1, a1 = 20 * 1
        # size: delta_1 = 20 * 1
        delta_1 = (w2.transpose() @ delta_2) * (a1 * (1 - a1))

        # update
        # size: w3 = 9 * 26, delta_L = 9 * 1, a2 = 26 * 1
        # size: b3 = 9 * 1, delta_L = 9 * 1
        w3 = w3 - learning_rate * (delta_L @ a2.transpose())
        b3 = b3 - learning_rate * delta_L

        # size: w2 = 26 * 20, delta_2 = 26 * 1, a1 = 20 * 1
        # size: b2 = 26 * 1, delta_2 = 26 * 1
        w2 = w2 - learning_rate * (delta_2 @ a1.transpose())
        b2 = b2 - learning_rate * delta_2

        # size: w1 = 20 * 784, delta_1 = 20 * 1, x_tmp = 1 * 784
        # size: b1 = 20 * 1, delta_1 = 20 * 1
        w1 = w1 - learning_rate * (delta_1 @ x_tmp)
        b1 = b1 - learning_rate * delta_1

    # calcualte accuracy rate for not overfitting ==================

    # training data 
    training_correct_num = 0
    for j in range (len(training_data_x)):

        x_tmp = []
        x_tmp.append(training_data_x[j])
        x_tmp = np.array(x_tmp)

        y_tmp = []
        y_tmp.append(training_data_y[j])
        y_tmp = np.array(y_tmp)

        n1 = w1 @ x_tmp.transpose() + b1
        a1 = Sigmoid(n1)

        n2 = w2 @ a1 + b2
        a2 = Sigmoid(n2)

        n3 = w3 @ a2 + b3
        aL = Softmax(n3)

        # caculate correct num 

        max_value_aL = -2
        max_idx_aL = 0
        for k in range (len(aL)):
            if(max_value_aL < aL[k][0]):
                max_value_aL = aL[k][0]
                max_idx_aL = k

        max_idx_y = 0
        for k in range (len(y_tmp[0])):
            if(y_tmp[0][k] == 1):
                max_idx_y = k
                break
        
        if(max_idx_aL == max_idx_y):
            training_correct_num += 1

    # verify data -----------------------------
    verify_correct_num = 0

    for j in range (len(verify_data_x)):

        x_tmp = []
        x_tmp.append(verify_data_x[j])
        x_tmp = np.array(x_tmp)

        y_tmp = []
        y_tmp.append(verify_data_y[j])
        y_tmp = np.array(y_tmp)

        n1 = w1 @ x_tmp.transpose() + b1
        a1 = Sigmoid(n1)

        n2 = w2 @ a1 + b2
        a2 = Sigmoid(n2)

        n3 = w3 @ a2 + b3
        aL = Softmax(n3)

        # caculate correct num

        max_value_aL = -2
        max_idx_aL = 0
        for k in range (len(aL)):
            if(max_value_aL < aL[k][0]):
                max_value_aL = aL[k][0]
                max_idx_aL = k

        max_idx_y = 0
        for k in range (len(y_tmp[0])):
            if(y_tmp[0][k] == 1):
                max_idx_y = k
                break
        
        if(max_idx_aL == max_idx_y):
            verify_correct_num += 1

    training_correct_rate = training_correct_num / len(training_data_x)
    verify_correct_rate = verify_correct_num / len(verify_data_x)

    # avoid overfitting
    if(training_correct_rate > 0.96 or verify_correct_rate > 0.96):
        epoch_result = i+1
        break

    # output information for drawing compare plot
    output_epoch.append(i+1)
    output_training_correct_rate.append(training_correct_rate)
    output_verify_correct_rate.append(verify_correct_rate)

    # print("Epoch now:", i+1)
    # print("Training Data Correct rate:", training_correct_num / len(training_data_x))
    # print("Verify Data Correct rate:"  , verify_correct_num / (len(verify_data_x)))
    # print()

"""
# draw the compare plot ===================================================

plt.plot(output_epoch, output_training_correct_rate, color='indianred')
plt.plot(output_epoch, output_verify_correct_rate, color='#7eb54e')

# str_title = "Training Correct Rate by Learning Rate " + str(learning_rate)
plt.title("Training Correct Rate vs Verify Correct Rate" )

plt.xlabel("Epoch")
plt.ylabel("Correct Rate")

plt.legend(["Training Correct Rate", "Verify Correct Rate"], loc = "lower right")
plt.grid()
plt.savefig('output9.png')
plt.show()

"""

# Predict Test9 ============================================================

df = pd.read_csv('mnist_test9.csv')

# row size: 9020, col size: 784
rowx, colx = df.shape

test_data_x = df.to_numpy()[:, :] / 1000

classification = [1, 2, 3, 4, 5, 6, 7, 8, 9]

ans_result = []
for i in range (len(test_data_x)):
    x_tmp = []
    x_tmp.append(test_data_x[i])
    x_tmp = np.array(x_tmp)

    n1 = w1 @ x_tmp.transpose() + b1
    a1 = Sigmoid(n1)

    n2 = w2 @ a1 + b2
    a2 = Sigmoid(n2)

    n3 = w3 @ a2 + b3
    aL = Softmax(n3)

    aL_idx_classification = np.argmax(aL)
    ans_result.append(classification[aL_idx_classification])
    # print(aL_classification, classification[aL_classification])

ans_result = np.array(ans_result)
ans_result_df = pd.DataFrame(ans_result)
ans_result_df.to_csv('ans9.csv', index=False, header=False)

# output result
hidden_layer_neurons = [w1.shape[0], w2.shape[0], w3.shape[0]]
print("Final Result:")
print("Epoch:", epoch_result, ", Learning Rate:", learning_rate, ", Hidden Layer:", hidden_layer_neurons)
print("Training Data Correct Rate:", training_correct_num / len(training_data_x))
print("Validation Data Correct Rate:"  , verify_correct_num / (len(verify_data_x)))
    
