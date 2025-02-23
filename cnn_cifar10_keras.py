# Train a very deep Convolutional Neural Network for classification of CIFAR10 small images
# using keras with tensorflow backend.
# Many thanks to:
# Very Deep Convolutional Networks for Large-Scale Image Recognition, K. Simonyan, A. Zisserman, 2014
# cifar-10: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
# TensorFlow: Large-scale machine learning on heterogeneous systems
# keras: https://github.com/fchollet/keras
# scikit-learn: http://scikit-learn.org/stable/index.html
# follow the logic flow of a Jupyter Notebook
# http://nbviewer.jupyter.org/github/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from scipy.misc import imsave
from keras.utils.visualize_util import plot
from keras.models import model_from_json
from util_functions import nice_imshow, make_mosaic
from sklearn.metrics import classification_report
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.preprocessing import label_binarize

import tensorflow as tf
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import time
import pickle as pickle
np.random.seed(1337)  # for reproducibility

batch_size = 32
nb_classes = 10
nb_epoch = 50
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
X_val = X_train[40000:50000]
y_val = y_train[40000:50000]
X_train = X_train[0:40000]
y_train = y_train[0:40000]

print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')
# print(X_train.dtype)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# define model structure
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])
plot(model, to_file='cnn_model.png')

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255

# load models
# model = model_from_json(open('model.json').read())

# load weights or train the model:
WEIGHTS_FNAME = 'weights.h5'
if True and os.path.exists(WEIGHTS_FNAME):
    # Just change the True to false to force re-training
    model.load_weights(WEIGHTS_FNAME)
    print('Existing weights loaded.')
else:
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch, show_accuracy=True,
                  validation_data=(X_test, Y_test), shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
        # fit the model on the batches generated by datagen.flow()
        hist = model.fit_generator(datagen.flow(X_train, Y_train,
                                                batch_size=batch_size),
                                   samples_per_epoch=X_train.shape[0],
                                   nb_epoch=nb_epoch, show_accuracy=True,
                                   validation_data=(X_val, Y_val),
                                   nb_worker=1)

        print(hist.history)
    # save as JSON
    json_string = model.to_json()
    model = model_from_json(json_string)
    model.save_weights('my_model_weights.h5')
    print('model weights saved.')

# visualize the imput image
im_index = 5000
X = X_train[im_index:im_index + 1]
# print(X.shape)
X = np.swapaxes(X, 1, 3)
X = np.swapaxes(X, 1, 2)
plt.figure(figsize=(10, 10))
plt.suptitle('input image')
plt.imshow(np.squeeze(X), interpolation='none')
# plt.show()

layer_idx = 2
# Visualize weights
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
W = sess.run(model.layers[layer_idx].W)
print("weight shape : ", W.shape)

fig, axs = plt.subplots(6, 6, figsize=(10, 10))
axs = axs.ravel()
for i in range(6 * 6):
    if i < W.shape[0]:
        axs[i].imshow(W[i, 0, :, :], interpolation='none', cmap='Greys_r')
#    axs[i].set_axis_off()
    axs[i].axis('off')
plt.suptitle('relu1 weights')
# plt.show()

# visualize the output layers
convout1_f = K.function([model.layers[0].input], [
                        model.layers[layer_idx].output])
X = X_train[im_index:im_index + 1]
C1 = convout1_f([X])[0]
C1 = np.squeeze(C1)
# print("C1 shape : ", C1.shape)

plt.figure(figsize=(10, 10))
plt.suptitle('relu1 output')
nice_imshow(plt.gca(), make_mosaic(C1, 6, 6), cmap=cm.binary)
# plt.subplot(6,6,2); plt.axis('off'); plt.imshow(C1[0,:,:], cmap=cm.binary)
# plt.show()

# Calculate test set score with Keras
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Calculate test set score manually after prediction
Y_pred = model.predict(X_test)
# Convert one-hot to index
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

print(classification_report(y_test, y_pred))

predict = model.predict_proba(X_test, batch_size=batch_size, verbose=1)
# print(predict)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
y = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
for i in range(nb_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), predict.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

##############################################################################
# Plot of a ROC curve for a specific class
# plt.figure()
# plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

##############################################################################
# Plot ROC curves for the multiclass problem
# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(nb_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= nb_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         linewidth=2)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         linewidth=2)
plot_name = 'no variation'
pickle.dump(roc_auc, open('roc_auc_' + plot_name + '.p', 'wb'))
pickle.dump(fpr, open('fpr_' + plot_name + '.p', 'wb'))
pickle.dump(tpr, open('tpr_' + plot_name + '.p', 'wb'))
for i in range(nb_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# Plot Confusion matrix
category_names = ['airplane', 'automobile', 'bird', 'cat',
                  'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category_names))
    plt.xticks(tick_marks, category_names, rotation=45)
    plt.yticks(tick_marks, category_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_confusion_matrix(y_test, y_pred, flag_plot, plot_title):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm)

    # Normalize the confusion matrix by row (i.e by the number of samples in
    # each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(
        cm_normalized, title='Normalized confusion matrix for ' + plot_title)
    if flag_plot == 1:
        plt.savefig('Confusion_Matrix_' + plot_name +
                    '.png', bbox_inches='tight')
    diag_sum = 0
    for i in range(6):
        diag_sum += cm_normalized[i][i]
    cm_normalized_sum = sum(sum(cm_normalized))
    accuracy = diag_sum / cm_normalized_sum
    return (cm, cm_normalized, accuracy)

result = compute_confusion_matrix(y_test, y_pred, 1, plot_name)
