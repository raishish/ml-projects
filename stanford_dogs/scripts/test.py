import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import time

# to get reproducible results
np.random.seed(22)

# config
img_width, img_height = 224, 224
batch_size = 32
model_path = '../models/model.hdf5'
test_data_dir = '../data/test_dataset'

def test():
    model = load_model(model_path)

    datagen = ImageDataGenerator(rescale=1./255)

    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    scores = model.evaluate_generator(test_generator, steps = batch_size)
    print "Accuracy:", scores[1]

if __name__ == '__main__':
    # test the accuracy of the model on a test dataset
    test()