import numpy as np
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf

# 模型路径
model_save_path = './alexnet8_training_model/traning_mnist_model.ckpt'
print('-------------load the model-----------------')
# 定义模型结构 lenet5
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5),activation='sigmoid',input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
#     tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5),activation='sigmoid'),
#     tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
#     # tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(84, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#alexnet
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5),input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),#《Batch Normalization: Accelerating Deep Network Training by  Reducing Internal Covariate Shift》
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same',activation='relu'),
    # tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation='sigmoid'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(2048, activation='sigmoid'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(10, activation='softmax')
])

print('compile')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy', 'mse','mae',])
model.load_weights(model_save_path)

# 画前5个灰度图
def plot_example(X, y):
    """Plot the first 5 images and their labels in a row."""
    for i, (img, y) in enumerate(zip(X[:5].reshape(5, 28, 28), y[:5])):
        plt.subplot(151 + i)
        plt.imshow(img.T, cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.title(y)
    plt.show()

if __name__ == '__main__':
    # 加载数据
    annots = loadmat('./challenge/cdigits.mat')
    digits_vec = annots['cdigits_vec']
    digits_lables = annots['cdigits_labels']
    testX = digits_vec.T
    testY = digits_lables[0]
    mnist_dim = testX.shape[0]

    # 计算正确预测的数量
    correct_num = 0
    for i, (img, y) in enumerate(zip(testX[:mnist_dim].reshape(mnist_dim, 28, 28), testY[:mnist_dim])):
        # 将np形式的array转换成灰度图，再转回np的array
        img_arr = np.array(Image.fromarray(img.T, 'L'))
        img_arr = img_arr.reshape(28, 28, 1)
        img_arr = img_arr.astype('float32')
        img_arr = img_arr / 255.0
        x_predict = img_arr[tf.newaxis, ...]
        # 模型最后输出层的概率，每次都在变，说明我这个循环逻辑应该没问题
        result = model.predict(x_predict)
        # 找最大的可能性的label
        pred = tf.argmax(result, axis=1)
        # 如果预测正确,correct_num + 1
        if np.array(pred[0] == y):
            correct_num += 1
        print('\n')
        print('The handwriting number is: ', np.array(pred)[0])
        print('The lable is: ', y)

    # 除以图片总数计算概率
    accuarcy = correct_num / mnist_dim
    print('accuracy: ' + str(accuarcy))

    # # 画前5张灰度图
    # # plot data sample
    # plot_example(testX, testY)



