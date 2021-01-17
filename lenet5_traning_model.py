import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
np.set_printoptions(threshold=np.inf)
mnist_dataset = tf.keras.datasets.mnist
(training_data, training_label), (val_data, val_label) = mnist_dataset.load_data()
training_data = training_data.reshape(training_data.shape[0],28,28,1)
val_data = val_data.reshape(val_data.shape[0],28,28,1)

digits_preprocessing = ImageDataGenerator(
    rescale=1. / 1.,  
    rotation_range=90,  
    width_shift_range=.15,  
    height_shift_range=.15,  
    horizontal_flip=True,  
    zoom_range=0.3  
)
digits_preprocessing.fit(training_data)

training_data = training_data.astype('float32')
val_data = val_data.astype('float32')
training_data= training_data / 255.0
val_data= val_data / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5),activation='sigmoid',input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5),activation='sigmoid'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy', 'mse','mae',])

training_model_path = "./lenet5_training_model/traning_mnist_model.ckpt"
if os.path.exists(training_model_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(training_model_path)

model_record = tf.keras.callbacks.ModelCheckpoint(filepath=training_model_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

recoeding = model.fit(training_data, training_label, batch_size=32, epochs=5, validation_data=(val_data, val_label), validation_freq=1,verbose=1,
                    callbacks=[model_record])
model.summary()

# acc and loss
training_acc = recoeding.history['sparse_categorical_accuracy']
val_acc = recoeding.history['val_sparse_categorical_accuracy']
training_loss = recoeding.history['loss']
val_loss = recoeding.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(training_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
