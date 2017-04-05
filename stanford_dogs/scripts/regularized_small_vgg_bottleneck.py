import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D, Input
from keras import applications, optimizers,regularizers
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K

# dealing with NaN in case of large gradients
# can't set this since GPU driver does not support it
# K.set_floatx('float64')

import matplotlib
matplotlib.use('Agg')       # since woking on a remote server
# for plotting plots
from matplotlib import pyplot as plt
import time

# to get reproducible results
np.random.seed(22)

# config
img_width, img_height = 224, 224
train_data_dir = '../data/keras_small_ds/train'
validation_data_dir = '../data/keras_small_ds/validation'
nb_train_samples = 8400
nb_validation_samples = 3600
epochs = 20
batch_size = 64
nb_classes = 120
samples_per_batch = 1200
# bottleneck_weights_path = "weights/reg_bottleneck_with_reg_conv_ep_20_model_weights.h5"
bottleneck_weights_path = "../models/weights/reg_2_final_model.h5"
# bottleneck_weights_path = "weights/smaller_bottleneck_model_weights.h5"
final_weights_path = "../models/weights/reg_3_final_model_weights.h5"
final_model_path = "../models/model.hdf5"

# plots
bottleneck_plot_path = '../plots/approach3/reg_vgg_conv_reg_rmsprop_lr_5e-04_step_decay' + '_ep_' + str(epochs) + '.png'
final_plot_path = '../plots/approach3/small_full_reg_2.png'


def train_bottleneck():
    input_tensor = Input(shape=(img_width, img_height, 3))

    base_model = applications.VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')   # laod vgg16 model with pretrained weights
    print "VGG16 model loaded."
    
    x = base_model.layers[14].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    predictions = Dense(120, activation='softmax')(x)

    model = Model(input = base_model.input, output = predictions)

    # first train the top fc layers
    # freeze all the layers of VGG16 
    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[11:14]:
        if not layer.__getattribute__('kernel_regularizer'):
            layer.__setattr__('kernel_regularizer', regularizers.l2(0.005))


    # learning rate schedule
    def step_decay(epoch):
        initial_lrate = 0.0005
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    rmsp = optimizers.rmsprop(lr = 0.0, decay = 0.0, rho = 0.9)

    print "Compiling bottleneck model..."
    model.compile(optimizer=rmsp,
                loss='categorical_crossentropy', metrics=['accuracy'])

    print "\n-------------Model layers-------------"
    print "Layer #", "\t", "Name", "\t\t", "Trainabe", "\t", "Regularization"
    for i, layer in enumerate(model.layers):
        print i, "\t", layer.name, "\t\t", layer.trainable, "\t",
        try:
            if layer.__getattribute__('kernel_regularizer'): print layer.__getattribute__('kernel_regularizer')
            else: print "None\n"
        except:
            print "\n"
    print "----------------------------------------------\n"


    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # callbacks
    early_stopping = EarlyStopping(monitor='val_acc', patience=4)
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate, early_stopping]
    # callbacks_list = [early_stopping]

    hist = model.fit_generator(train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks = callbacks_list)

    try:
        # save the model weights at this point
        print "Saving bottleneck model weigths..."
        model.save_weights(bottleneck_weights_path)
    except:
        print "Error while saving model..."

    # Plotting Losses and accuracies
    print "Plotting results..."
    train_loss = np.array(hist.history['loss'])
    train_acc = np.array(hist.history['acc'])
    val_loss = np.array(hist.history['val_loss'])
    val_acc = np.array(hist.history['val_acc'])

    epoch_no = np.array([i for i in xrange(1,len(train_loss) + 1)])

    try:
        plt.subplot(1,2,1)
        plt.plot(epoch_no, train_loss, 'r', label='train_loss')
        plt.plot(epoch_no, val_loss, 'b', label='val_loss')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(1,2,2)
        plt.plot(epoch_no, train_acc * 100, 'g',label='train_acc')
        plt.plot(epoch_no, val_acc * 100, 'm', label='val_acc')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')

        title = "Bottleneck VGG16 FC Model\n" + "[classifier: {}, ".format(fc_act.get_config()['activation']) + \
                "Optimizer: RMSPROP, lr: {}, rho: {:.4f}, decay: {}]".format( 5e-04,
                rmsp.get_config()['rho'], '0.5/10 epochs')

        plt.suptitle(title)
        plt.subplots_adjust(top=0.8)

        plt.savefig(bottleneck_plot_path)
    except:
        print "Plotting error"

    max_train_acc = max(hist.history['acc'])
    max_test_acc = max(hist.history['val_acc'])

    results = '\nResults:\n\n' + \
        'Max train_acc of  {}  at Epoch  #{}\n'.format(max_train_acc, hist.history['acc'].index(max_train_acc) + 1) + \
        'Final train_acc of  {}  at Epoch  #{}\n'.format(hist.history['acc'][-1], len(train_acc)) + \
        'Max test_acc of  {}  at Epoch   #{}\n'.format(max_test_acc, hist.history['val_acc'].index(max_test_acc) + 1) + \
        'Final test_acc of  {}  at Epoch  #{}\n'.format(hist.history['val_acc'][-1], len(val_acc)) + \
        "--------------------------------------------------------------------------\n"

    changes = "\n--------------------------------------------------------------------------\n" + \
    "Implementation Changes:\n\n" + \
    "reduced the number of layers, removed the last 3 conv and 1 max pool layers\n" + \
    "added l2 regularization (lambda = 0.01) to the last dense (relu) layer" + \
    "as well as last 3 conv layers (lambda = 0.005)" + \
    "added l2 regularization (lambda = 0.01) to the last dense (relu) layer" + \
    "Also reduced the lr = 5e-04 with step decay of 0.5 every 10 epochs" + \
    "optimizer: RMSprop lr 5e-04\n"

    try:
        with open('logs.txt', 'a') as logf:
            logf.write(changes + results)
    except:
        print "File write error: logs.txt"


def fine_tune():
    print "Fine tuning the model with 10 epochs and increased regularization..."
    input_tensor = Input(shape=(img_width, img_height, 3))

    base_model = applications.VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')   # laod vgg16 model with pretrained weights
    print "VGG16 model loaded."
    
    x = base_model.layers[14].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.05))(x)
    predictions = Dense(120, activation='softmax')(x)

    model = Model(input = base_model.input, output = predictions)
    model.load_weights(bottleneck_weights_path)

    # Now train the last 6 convolutional layers and freeze the rest
    for layer in model.layers[:7]:
        layer.trainable = False
    for layer in model.layers[7:]:
        layer.trainable = True

    for layer in base_model.layers[7:10]:
        if not layer.__getattribute__('kernel_regularizer'):
            layer.__setattr__('kernel_regularizer', regularizers.l2(0.005))
    for layer in base_model.layers[11:14]:
        if not layer.__getattribute__('kernel_regularizer'):
            layer.__setattr__('kernel_regularizer', regularizers.l2(0.01))


    sgd = optimizers.SGD(lr=0.0, momentum=0.9)      # since we have a custom lr scheduler
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy', metrics=['accuracy'])

    print "\n-------------Model layers-------------"
    print "Layer #", "\t", "Name", "\t\t", "Trainabe", "\t", "Regularization"
    for i, layer in enumerate(model.layers):
        print i, "\t", layer.name, "\t\t", layer.trainable, "\t",
        try:
            if layer.__getattribute__('kernel_regularizer'): print layer.__getattribute__('kernel_regularizer')
            else: print "None\n"
        except:
            print "\n"
    print "----------------------------------------------\n"

    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # learning rate schedule
    def step_decay(epoch):
        initial_lrate = 0.0001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    # callbacks
    early_stopping = EarlyStopping(monitor='val_acc', patience=4)
    lrate = LearningRateScheduler(step_decay)
    model_checkpoint = ModelCheckpoint(filepath = 'model.{epoch:02d}-{val_acc:.2f}.hdf5', period = 4, verbose = 1)
    callbacks_list = [early_stopping, lrate, model_checkpoint]

    print "Initial learning rate:", sgd.get_config()['lr']

    hist = model.fit_generator(train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks = callbacks_list)

    try:
        # saving weights
        print "Saving final model weigths..."
        model.save_weights(final_weights_path)
        # saving model
        print "Saving final model..."
        model.save(final_model_path)
    except:
        print "Error saving model..."

    # plotting accuracies and losses
    print "Plotting..."
    train_loss = np.array(hist.history['loss'])
    train_acc = np.array(hist.history['acc'])
    val_loss = np.array(hist.history['val_loss'])
    val_acc = np.array(hist.history['val_acc'])

    epoch_no = np.array([i for i in xrange(1,len(train_loss) + 1)])

    try:
        plt.subplot(1,2,1)
        plt.plot(epoch_no, train_loss, 'r', label='train_loss')
        plt.plot(epoch_no, val_loss, 'b', label='val_loss')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(1,2,2)
        plt.plot(epoch_no, train_acc * 100, 'g',label='train_acc')
        plt.plot(epoch_no, val_acc * 100, 'm', label='val_acc')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')

        title = "VGG16 FC Model (fine-tuning)\n" + "[classifier: {}, ".format('softmax') + \
            "Optimizer: SGD, lr: {:.4f}, mom: {:.4f}]".format( sgd.get_config()['lr'],
                sgd.get_config()['momentum'])

        plt.suptitle(title)
        plt.subplots_adjust(top=0.8)

        plt.savefig(final_plot_path)
   #     plt.show()
    except:
            print "Plotting error"

    max_train_acc = max(hist.history['acc'])
    max_test_acc = max(hist.history['val_acc'])
    
    results = '\nResults:\n\n' + \
        'Max train_acc of  {}  at Epoch  #{}\n'.format(max_train_acc, hist.history['acc'].index(max_train_acc) + 1) + \
        'Final train_acc of  {}  at Epoch  #{}\n'.format(hist.history['acc'][-1], len(train_acc)) + \
        'Max test_acc of  {}  at Epoch   #{}\n'.format(max_test_acc, hist.history['val_acc'].index(max_test_acc) + 1) + \
        'Final test_acc of  {}  at Epoch  #{}\n'.format(hist.history['val_acc'][-1], len(val_acc)) + \
        "--------------------------------------------------------------------------\n"

    changes = "\n--------------------------------------------------------------------------\n" + \
        "Fine tuning the model:\n\n" + \
        "training last 6 conv layers\n" + \
        "regularization: last dense (lambda = 0.05), last 3 conv (lambda = 0.01), next 3 conv above them (lambda = 0.005)\n" + \
        "with SGD with lr = 1e-04, step_decay = 0.5 / 10 epochs and momentum = 0.9\n"


    try:
        with open('logs_arya.txt', 'a') as logf:
            logf.write(changes + results)
    except:
        print "File write error: logs.txt"


if __name__ == '__main__':
    # train_bottleneck()        # train the bottleneck model
    fine_tune()         # fine tune the model
