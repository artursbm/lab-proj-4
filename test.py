import cv2
import sys
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import argparse
import imutils
import pickle
import os

def update_dashes(secret, cur_dash, rec_guess):
  result = ""
  
  for i in range(len(secret)):
    if secret[i] == rec_guess:
      result = result + rec_guess     # Adds guess to string if guess is correctly
      
    else:
      # Add the dash at index i to result if it doesn't match the guess
      result = result + cur_dash[i]
      
  return result


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True, help="path to label binarizer")
args = vars(ap.parse_args())

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

video_capture = cv2.VideoCapture(0)
cv2.namedWindow('Model Image')

# set the ration of main video screen
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# VARIABLES INITIALIZATION
# THRESHOLD - ratio of the same letter in the last N_FRAMES predicted letters
THRESHOLD = 0.85
N_FRAMES = 25

IMG_SIZE = 96
SENTENCE = '' # string that will hold the final output
letter = '' # temporary letter
LETTERS = np.array([], dtype='object') # array with predicted letters

START = False # start/pause controller
GUESS = False
# supportive text
description_text_1 = "Press 'S' to start/pause gesture recognition."
description_text_2 = "Press 'C' to capture your guess."
description_text_3 = "Press 'R' to restart the game. "
description_text_4 = "Press 'Q' to quit."

##### HANGMAN CODE #####
#here we set the secret
WORD = "secret"
#creates an variable with an empty value
GUESSES = ''
#determine the number of turns
TURNS = 10
DASHES = "-" * len(WORD)

game_text = 'You have {} guesses.'.format(TURNS)

while True:
    blank_image = np.zeros((200,800,3), np.uint8) # black image for the output
    ret, frame = video_capture.read() # capture frame-by-frame
    # set the corners for the square to initialize the model picture frame
    x_0 = int(frame.shape[1] * 0.1)
    y_0 = int(frame.shape[0] * 0.25)
    x_1 = int(x_0 + 200)
    y_1 = int(y_0 + 200)

    # MODEL IMAGE INITIALIZATION
    hand = frame.copy()[y_0:y_1, x_0:x_1] # crop model image
    gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blured = cv2.GaussianBlur(gray, (5, 5), 0) # noise reduction
    
    model_image = blured
    model_image = cv2.resize(model_image, (IMG_SIZE, IMG_SIZE))
    model_image = model_image.astype("float") / 255.0
    model_image = img_to_array(model_image)
    model_image = np.expand_dims(model_image, axis=0)

    try:
        predict = model.predict(model_image)
        for values in predict:
            if np.all(values < 0.5):
                # if probability of each class is less than .5 return a message
                letter = 'Cannot classify :('
            else:
                proba = predict[0]
                idx = np.argmax(proba)
                letter = lb.classes_[idx]
                LETTERS = np.append(LETTERS, letter)
    except:
        pass


    if START == True:
        if GUESS == True and TURNS > 0 and not DASHES == WORD:
            TURNS = TURNS - 1
            GUESS = False
            if (np.mean(LETTERS[-N_FRAMES:] == letter) >= THRESHOLD) & (len(LETTERS) >= N_FRAMES):
                if letter in WORD:
                    DASHES = update_dashes(WORD, DASHES, letter)
                    game_text = 'You guessed right!'
                else:
                    game_text = 'Wrong! You have {} guesses left.'.format(TURNS)

    if DASHES == WORD:
        game_text('You won!')


    if START == False:
        paused_text = 'Paused'
    else:
        paused_text = ''

    # TEXT INITIALIZATION
    # paaused text
    cv2.putText(
        img=frame,
        text=paused_text,
        org=(x_0+140,y_0+195),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(0,0,255),
        fontScale=1
    )

    # helper texts
    cv2.putText(
        img=frame,
        text=description_text_1,
        org=(10,430),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255,255,255),
        fontScale=1
    )

    cv2.putText(
        img=frame,
        text=description_text_2,
        org=(10,445),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255,255,255),
        fontScale=1
    )

    cv2.putText(
        img=frame,
        text=description_text_3,
        org=(10,460),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255,255,255),
        fontScale=1
    )

    cv2.putText(
        img=frame,
        text=description_text_4,
        org=(10,475),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255,255,255),
        fontScale=1
    )

    cv2.putText(
        img=frame,
        text='Place your hand here:',
        org=(x_0-30,y_0-10),
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        color=(255,255,255),
        fontScale=1
    )

    # current letter
    cv2.putText(
        img=frame,
        text=letter,
        org=(x_0+10,y_0+20),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255,255,255),
        fontScale=1
    )

    # final output
    cv2.putText(
        img=blank_image,
        text='Result: ' + DASHES,
        org=(10, 50),
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        thickness=1,
        color=(0,0,255),
        fontScale=1
    )

    # final output
    cv2.putText(
        img=blank_image,
        text=game_text,
        org=(10, 100),
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        thickness=1,
        color=(0,255,0),
        fontScale=1
    )

    # draw rectangle for hand placement
    cv2.rectangle(frame, (x_0, y_0), (x_1, y_1), (0, 255, 0), 2)

    # display the resulting frames
    frame = cv2.flip(frame, 0)
    blured = cv2.flip(blured, 0)
    cv2.imshow('Main Image', frame)
    cv2.imshow('Model Image', blured)
    cv2.imshow('Output', blank_image)

    if cv2.waitKey(10) & 0xFF == ord('s'):
        START = not START

    if cv2.waitKey(10) & 0xFF == ord('c'):
        GUESS = True

    if cv2.waitKey(10) & 0xFF == ord('r'):
        SENTENCE = ''
        TURNS = 10
        DASHES = "-" * len(WORD)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()