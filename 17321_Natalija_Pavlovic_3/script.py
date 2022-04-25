import numpy as np
import cv2 as cv
import dlib


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


faceDetector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

mona = cv.imread("mona.jpg")
gray = cv.cvtColor(mona, cv.COLOR_BGR2GRAY)

rects = dlib.rectangles()
faces = faceDetector.detectMultiScale(gray, 1.8, 5)

for index, (x, y, w, h) in enumerate(faces):
    cv.rectangle(mona, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.putText(mona, text="Face", org=(x, y-5), fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=0.5,
               color=(0, 0, 255), thickness=2)
    rects.append(dlib.rectangle(x, y, x + w, y + h))

for rect in rects:
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    # lice
    a = np.append(shape[0:17], np.flip(shape[17:27], axis=0), axis=0)
    cv.fillPoly(mona, [a], (0, 255, 0))
    cv.polylines(mona, [a], True, (0, 255, 255), 2)

    # nos
    a = shape[27:31]
    cv.polylines(mona, [a], True, (0, 0, 255), 2)
    a = shape[30:36]
    cv.fillPoly(mona, [a], (255, 0, 255))
    cv.polylines(mona, [a], True, (0, 0, 255), 2)

    # usta
    # gornja
    a = np.append(shape[48:55], np.flip(shape[60:65], axis=0), axis=0)
    cv.fillPoly(mona, [a], (255, 255, 51))
    cv.polylines(mona, [a], True, (0, 255, 255), 2)
    # donja
    a = shape[54:65]
    cv.fillPoly(mona, [a], (0, 0, 255))
    cv.polylines(mona, [a], True, (255, 0, 255), 2)

    # levo oko
    a = np.append(shape[17:22], np.flip(shape[36:40], axis=0), axis=0)
    cv.fillPoly(mona, [a], (255, 0, 0))
    cv.polylines(mona, [a], True, (255, 0, 0), 2)

    a = shape[36:42]
    cv.fillPoly(mona, [a], (0, 0, 255))
    cv.polylines(mona, [a], True, (255, 0, 0), 2)

    # desno oko
    a = np.append(shape[22:27], np.flip(shape[42:46], axis=0), axis=0)
    cv.fillPoly(mona, [a], (255, 0, 255))
    cv.polylines(mona, [a], True, (0, 255, 0), 2)

    a = shape[42:48]
    cv.fillPoly(mona, [a], (255, 0, 255))
    cv.polylines(mona, [a], True, (0, 255, 255), 2)

    for index, (x, y) in enumerate(shape):
        if (0 <= index < 27) or (46 <= index < 48):

            circle = cv.circle(mona, (x, y), 6, (128, 128, 128), thickness=-1)
        elif (40 <= index < 42) or (42 <= index < 46) or (31 <= index < 36):
            pt1 = (x, y - 8)
            pt2 = (x - 7, y + 7)
            pt3 = (x + 7, y + 7)
            trianglePoints = np.array([pt1, pt2, pt3])
            cv.drawContours(mona, [trianglePoints], 0, (128, 128, 128), -1)
        else:
            cv.rectangle(mona, (x + 5, y + 5), (x - 5, y - 5), (128, 128, 128), -1)

cv.imshow("output", mona)
cv.imwrite("output.jpg", mona)

cv.waitKey(0)
cv.destroyAllWindows()
