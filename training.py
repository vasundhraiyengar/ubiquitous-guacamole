import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# dimensions of our images.
img_width, img_height = 50,50

train_data_dir = 'data'
nb_train_samples = 500
epochs = 50
batch_size = 1


json_file = open('gesture_detection_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#model.load_weights('gesture_detection_model_weights.h5')

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


datagen = ImageDataGenerator(rescale=1./255)


generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

checkpoint = ModelCheckpoint('gesture_detection_model_weights.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit_generator(generator, epochs=epochs, callbacks=callbacks_list)
scores = model.evaluate_generator(generator)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
# serialize weights to HDF5
#model.save_weights("gesture_detection_model_weights.h5")
#print("Saved weights to disk")
