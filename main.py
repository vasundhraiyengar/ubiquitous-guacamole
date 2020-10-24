from PIL import Image
from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import math
import os
import numpy as np
from keras import Sequential
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array ,
load_img
from pygame import mixer
import time
capture = cv2.VideoCapture(0)
# Loading Hand Detection Model - Inference Graph
graph, session = detector_utils.load_inference_graph()
# Loading Gesture Detection Model
json_file = open ( ’ gesture_detection_model . json ’ , ’r ’)
loaded_model_json = json_file.read()
json_file . close ()
model = model_from_json(loaded_model_json)
# Loading weights into gesture model
model.load_weights("gesture_detection_model_weights_col.
h5")
print ("Loaded gesture model from disk")
# Compiling Gesture Model
model . compile (loss=’categorical_crossentropy’,
optimizer=’sgd’, metrics=[’accuracy’])
if __name__ == ’__main__’:
# Arguments to OpenCV
parser = argparse.ArgumentParser()
parser . add_argument ( ’-sth ’ , ’--scorethreshold ’ ,
dest=’ score_thresh ’ , type=float ,
default =0.5)
parser . add_argument ( ’-fps ’ , ’--fps ’ , dest=’fps ’ , type
=int ,
default=1)
parser . add_argument ( ’-src ’ , ’--source ’ , dest=’
video_source ’ ,
default=0)
parser . add_argument ( ’-wd ’ , ’--width ’ , dest=’width ’ ,
type=int ,
default=640)
parser . add_argument ( ’-ht ’ , ’--height ’ , dest=’ height ’ ,
type=int ,
default=480)
parser . add_argument ( ’-ds ’ , ’--display ’ ,
dest=’ display ’ , type=int ,
default=1)
parser . add_argument ( ’-num-w’, ’--num-workers ’ ,
dest=’num_workers ’ , type=int ,
default=4)
parser . add_argument ( ’-q-size ’ , ’--queue-size ’ ,
dest=’ queue_size ’ , type=int ,
default=5)
args = parser.parse_args()
# Video Capture
capture = cv2.VideoCapture(args.video_source)
capture . set ( cv2 . CAP_PROP_FRAME_WIDTH , a r g s . w i d t h )
capture . set ( cv2 . CAP_PROP_FRAME_HEIGHT , a r g s . h e i g h t )
# FPS
start_time = datetime.datetime.now()
num_frames = 0
# Size of video to be displayed through OpenCV
im_width, image_height = (capture.get(3), capture.get
(4))
# Number of hands to be detected
num_hands_detect = 1
#count = 1
prev_prediction = -1
chord = ’’
while True :
# Read Image
ret , image_from_cv = capture.read()
image_from_cv = cv2.flip(image_from_cv, 1)
image_from_cv = cv2.cvtColor(image_from_cv,
cv2.COLOR_BGR2RGB)
# Get bounding boxes from hand detection model
boxes, scores = detector_utils.detect_objects(
image_from_cv, graph, session)
# Draw bounding boxes
detector_utils.draw_box_on_image(
num_hands_detect , args.score_thresh ,
scores , boxes , im_width , image_height ,
image_from_cv)
# FPS
num_frames += 1
elapsed_time = (datetime.datetime.now() -
start_time ) . total_seconds ()
fps = num_frames / elapsed_time
if (args.display > 0):
if (args.fps > 0):
detector_utils.draw_fps_on_image(
"FPS : " + str ( int ( fps ) ) ,
image_from_cv)
cv2.putText(image_from_cv, chord, (100, 100),
cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255,
9) , 2)
cv2.imshow(’Real Time Gesture Detection’, cv2
. cvtColor (
image_from_cv , cv2 .COLOR_RGB2BGR) )
# To quit application press q
if cv2.waitKey(25) & 0xFF == ord (’q’):
cv2 . destroyAllWindows ()
break
else :print ("frames processed: ", num_frames,
" elapsed time : " , elapsed_time ,
" fps : " , str ( int (fps)))
for i in range ( num_hands_detect ) :
if (scores[i] > args.score_thresh):
(x1, x2, y1, y2) = (boxes[i][1] *
im_width , boxes [ i ][3] * im_width , boxes [ i ][0] * image_height , boxes [ i ][2] * image_height) x1 = int (math . floor (x1) )
y1 = int (math . floor (y1) )
x2 = int ( math . ceil ( x2 ) )
y2 = int ( math . ceil ( y2 ) )
#if cv2.waitKey(25) & 0xFF == ord(’c’):
#print("Image Captured")
# Crop hand from image and process
roi = image_from_cv[y1:y2, x1:x2]
roi = cv2 . cvtColor ( roi , cv2 .
COLOR_BGR2GRAY)
#( thresh , roi ) = cv2 . threshold ( roi ,
128, 255, cv2.THRESH_BINARY | cv2.
THRESH_OTSU)
#path = ’G:\Project\workspace\
handtracking ’
’’’
#Training
cv2.imwrite(os.path.join(path, ’Img’ + str(count) + ’.jpg’), roi)
img = Image.open(os.path.join(path,
’Img’ + str(count) + ’.jpg’))
img = img.resize((150,150), Image.
ANTIALIAS)
img . save ( os . path . join ( path , ’Img’ + str(count) + ’.jpg’))
count = count + 1;
’’’
img = Image.fromarray(roi)
img = img.resize((50,50), Image.ANTIALIAS)
img = img_to_array(img)
img = img / 255
img = np.expand_dims(img, axis = 0)
# Predict class of gesture
prob = model.predict(img)
prediction = prob.argmax(axis=-1)
# If accuracy greater than 90\%,
play chord and display
if prediction == 0 and prob . item (0) >=
0.9:
if prev_prediction == prediction:
break
mixer . init ()
mixer . music . load ( ’music /A.mp3’ )
mixer . music . play ()
print ( ’A’ )
chord = ’A Major’
elif prediction == 1 and prob . item (1) >=
0.9:
if prev_prediction == prediction:
break
mixer . init ()
mixer . music . load ( ’music /C.mp3’ )
mixer . music . play ()
print (’C’)
chord = ’C Major’
elif prediction == 2 and prob . item (2) >=
0.9:
if prev_prediction == prediction:
break
mixer . init ()
mixer . music . load ( ’music /D.mp3’ )
mixer . music . play ()
print ( ’D’ )
chord = ’D Major’
elif prediction == 3 and prob . item (3) >=
0.9:
if prev_prediction == prediction:
break
mixer . init ()
mixer . music . load ( ’ music /Em. mp3 ’ )
mixer . music . play ()
print ( ’Em’ )
chord = ’E Minor’
elif prediction == 4 and prob . item (4) >=
0.9:
if prev_prediction == prediction:
break
mixer . init ()
mixer . music . load ( ’music /G.mp3’ )
mixer . music . play ()
print ( ’G’ )
chord = ’G Major’
#break
prev_prediction = prediction
