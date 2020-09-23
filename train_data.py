import random
import cv2
import scipy

x_train = []
y_train = []

train_batch_pointer = 0
val_batch_pointer = 0

with open("./data/driving_dataset/data.txt") as f:
    for line in f:
        x_train.append("./data/driving_dataset/" + line.split()[0])
        y_train.append(float(line.split()[1]) * scipy.pi / 180)

num_images = len(x_train)

c = list(zip(x_train, y_train))
random.shuffle(c)
x_train, y_train = zip(*c)

train_x_train = x_train[:int(len(x_train) * 0.8)]
train_y_train = y_train[:int(len(x_train) * 0.8)]

val_x_train = x_train[-int(len(x_train) * 0.2):]
val_y_train = y_train[-int(len(x_train) * 0.2):]

num_train_images = len(train_x_train)
num_val_images = len(val_x_train)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(train_x_train[(train_batch_pointer + i) % num_train_images])[-150:], (120, 40),interpolation = cv2.INTER_AREA) / 255.0)
        y_out.append([train_y_train[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(val_x_train[(val_batch_pointer + i) % num_val_images])[-150:], (120, 40),interpolation = cv2.INTER_AREA) / 255.0)
        y_out.append([val_y_train[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
