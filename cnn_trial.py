# Following tutorial found
# https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense

# Initialize the CNN
classifier = Sequential()

classifier.add(Convolution2D(32,(3,3), input_shape = (64,64,3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

# Full Conection
classifier.add(Dense(activation = 'relu',units = 128))
classifier.add(Dense(activation = 'sigmoid',units = 1))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
# load data in
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(directory = 'data/dataset/',target_size = (64,64),
                                  batch_size = 32,
                                  class_mode = 'binary')

test_set = test_datagen.flow_from_directory(directory = 'data/dataset-test/',target_size = (64,64),
                                  batch_size = 32,
                                  class_mode = 'binary')
# Train network
from IPython.display import display
from PIL import Image
#train
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 800)
