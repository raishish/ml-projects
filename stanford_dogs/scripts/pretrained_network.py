import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras import applications, optimizers
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import backend as K

# dealing with NaN in case of large gradients
K.set_floatx('float64')

from matplotlib import pyplot as plt
# custom oneHot encoder
from extras import DataAug
import time

# to get reproducible results
np.random.seed(22)

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = '../data/keras_distributed_ds/train_7'
validation_data_2_dir = '../data/keras_distributed_ds/validation_2'
validation_data_3_dir = '../data/keras_distributed_ds/validation_3'
nb_train_samples = 8400
nb_validation_samples = 3600
epochs = 50
batch_size = 32
nb_classes = 120
samples_per_batch = 1200
top_model_path = '../model/bn_vgg16_sigmoid_var_lr_5e-04_epoch_' + str(epochs) + '_model' + '.h5'

# save bottleneck features
def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    print "Building VGG16 model...\t Time Elapsed: {}".format(time.time() - st_time)
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    # build the resnet50 network
    # model = applications.resnet50.ResNet50(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print "Forward pass started for training set #{}...\t Time Elapsed: {}".format(7, time.time() - st_time)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    print "Saving train features...\t Time Elapsed: {}".format(time.time() - st_time)
    np.save(open('features/bottleneck_vgg16_features_train_7.npy', 'wb'),
            bottleneck_features_train)

    print "Checking ..."
    t = np.load(open('../models/features/bottleneck_vgg16_features_train_7.npy', 'rb'))
    print "Shape of train_features:", t.shape

    generator = datagen.flow_from_directory(
        validation_data_2_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print "Forward pass started for validation #{} set...\t Time Elapsed: {}".format(3, time.time() - st_time)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)

    # plot_model(model, to_file='VGG16_model_{}_{}.png'.format(nb_train_samples, nb_validation_samples), show_shapes=True)

    print "Saving test features...\t Time Elapsed: {}".format(time.time() - st_time)
    np.save(open('../models/features/bottleneck_vgg16_features_validation_3.npy', 'wb'),
            bottleneck_features_validation)

    # save the model to carry on where you left
    # print "Saving the model...\t Time elapsed: {}".format(time.time() - st_time)
    # model.save('vgg16_model.h5')


def train_top_model():
    print "Training top layers..."
    train_data = np.load(open('features/bottleneck_vgg16_features_train.npy', 'rb'))
    train_labels = DataAug.oneHotEncoder_divided(nb_train_samples, nb_classes, nb_train_samples / samples_per_batch)

    validation_data = np.load(open('features/bottleneck_vgg16_features_validation.npy', 'rb'))
    validation_labels = DataAug.oneHotEncoder_divided(nb_validation_samples, nb_classes, nb_validation_samples / samples_per_batch)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.6))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))

    fc_act = Activation('sigmoid')
    # fc_act = Activation('softmax')
    model.add(Dense(nb_classes))
    model.add(fc_act)

    # learning rate schedule
    def step_decay(epoch):
        initial_lrate = 0.0005
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    # rmsp = optimizers.rmsprop(lr = 0.0006, decay=0.00001)
    # rmsp = optimizers.rmsprop(lr = 0.0005)      # since we now have defined a lr scheduler
    rmsp = optimizers.rmsprop(lr = 0.0, decay = 0.0, rho = 0.9)

    print "Compiling bottleneck model..."
    model.compile(optimizer=rmsp,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print "\n------------------------------------------------------------------------"
    print "Model Details:"
    print "Training data size:", train_data.shape[0], ", Validation data size:", validation_data.shape[0], ", Epochs:", epochs
    print "Final Layer Activation: {}".format(fc_act.get_config()['activation'])
    # print "Optimizer: RMSPROP, Learning Rate: {:.4f}, Rho: {:.4f}, Decay: {:.8f}".format(
        # rmsp.get_config()['lr'], rmsp.get_config()['rho'], rmsp.get_config()['decay'])
    print "Optimizer: RMSPROP, Learning Rate: {}, Rho: {:.4f}, Step decay: {}".format(
        '5e-04', rmsp.get_config()['rho'], '0.5')
    print "------------------------------------------------------------------------\n"

    # callbacks
    early_stopping = EarlyStopping(monitor='val_acc', patience=4)
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate, early_stopping]

    print "Training bottleneck model..."
    hist = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=callbacks_list)

    
    # Plotting Losses and accuracies
    train_loss = np.array(hist.history['loss'])
    train_acc = np.array(hist.history['acc'])
    val_loss = np.array(hist.history['val_loss'])
    val_acc = np.array(hist.history['val_acc'])

    epoch_no = np.array([i for i in xrange(1,len(train_loss) + 1)])

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
            "Optimizer: RMSPROP, lr: {:.4f}, rho: {:.4f}, decay: {}]".format( 5e-04,
                rmsp.get_config()['rho'], '0.5 per 10 epochs')

             # "Optimizer: RMSPROP, lr: {:.4f}, rho: {:.4f}, decay: {:.8f}]".format( rmsp.get_config()['lr'],
                # rmsp.get_config()['rho'], rmsp.get_config()['decay'])

    plt.suptitle(title)
    plt.subplots_adjust(top=0.8)
    
    # plt.savefig('../plots/bn_vgg16_sofmx_var_lr_' + str(rmsp.get_config()['lr']) + '_ep_' + str(epochs) + '.png')
    plt.savefig('../plots/approach2/bn_vgg16_sigmoid_var_lr_5e-04' + '_ep_' + str(epochs) + '.png')


    max_train_acc = max(hist.history['acc'])
    max_test_acc = max(hist.history['val_acc'])
    
    try:
        results = '\nResults:\n\n' + \
        'Max train_acc of  {}  at Epoch  #{}\n'.format(max_train_acc, hist.history['acc'].index(max_train_acc)) + \
        'Final train_acc of  {}  at Epoch  #{}\n'.format(hist.history['acc'][-1], len(train_acc)) + \
        'Max test_acc of  {}  at Epoch   #{}\n'.format(max_test_acc, hist.history['val_acc'].index(max_test_acc)) + \
        'Final test_acc of  {}  at Epoch  #{}\n'.format(hist.history['val_acc'][-1], len(val_acc)) + \
        "--------------------------------------------------------------------------\n"

    with open('logs.txt', 'a') as logf:
        logf.write(results)

    except:
        print "File write error: logs.txt"

    print "\nSaving model..."
    model.save(top_model_path)

    plt.show()



if __name__ == '__main__':
    changes = "\n--------------------------------------------------------------------------\n" + \
    "Implementation Changes:\n\n" + \
    "changed classifier activation back to sigmoid\n" + \
    "using constant lr 5e-04 w/ decay of 1e-05 for RMSprop\n" + \
    "saving the model this time.\n"

    with open('logs.txt', 'a') as logf:
        logf.write(changes)

    print "Using VGG16 as base network to improve upon."

    st_time = time.time()

    # save_bottleneck_features()
    train_top_model()

    print "\n--------------------------------------------------------"
    print "END!\nTotal time elapsed: {}".format(time.time() - st_time)
