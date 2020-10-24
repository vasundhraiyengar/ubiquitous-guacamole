import pydot
from keras.models import model_from_json
from keras.utils import plot_model

json_file = open('gesture_detection_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
plot_model(model, show_shapes=True, to_file='model.png')
print(model.summary())
