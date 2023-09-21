import cv2.cv2 as cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils
draw_specs = mp.solutions.hands.HAND_CONNECTIONS

# initialising sound control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

while True:
    success, img = vid.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        h, w, _ = img.shape

        coordinates = results.multi_hand_landmarks[0]
        draw.draw_landmarks(img, coordinates, draw_specs)

        # Getting cooridinates of point 4 and 8 in pixels
        x1, y1 = int(coordinates.landmark[4].x * w), int(coordinates.landmark[4].y * h)
        x2, y2 = int(coordinates.landmark[8].x * w), int(coordinates.landmark[8].y * h)

        cv2.circle(img, (x1, y1), 10, (255, 0, 9), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 9), cv2.FILLED)

        hyp = math.hypot(x2-x1, y2-y1)

        # Setting minimum length and maximum length
        MIN_LEN = 30
        MAX_LEN = 200
        if hyp < MIN_LEN:
            hyp = MIN_LEN
            cv2.circle(img, ((x1+x2)//2, (y1+y2)//2), 10, (9, 0, 255), cv2.FILLED)  # Red Dot if volume is muted

        if hyp > MAX_LEN:
            hyp = MAX_LEN
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 20), 3)  # Green Line if Volume is max
        else:
            cv2.line(img, (x1, y1), (x2, y2), (21, 160, 210), 3)  # Brown Line if Volume is not max

        minVol, maxVol, _ = volume.GetVolumeRange()

        # Converting the length range to volume range
        vol = (hyp - MIN_LEN)/(MAX_LEN - MIN_LEN) * (maxVol - minVol) + minVol
        volume.SetMasterVolumeLevel(vol, None)

    cv2.imshow("image", img)

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
