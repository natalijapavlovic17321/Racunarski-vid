import numpy as np
import time
import imutils
import cv2 as cv


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


def pyramid(image, scale=2, minSize=(180, 180)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

img = cv.imread("input.png")

x = 125
y = 20
w = 1440
h = 720
imgCropped = img[y:y + h, x:x + w]

(winW, winH) = (180, 180)

for resized in pyramid(imgCropped, scale=2):

    for (x, y, window) in sliding_window(resized, stepSize=180, windowSize=(winW, winH)):

        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        blob = cv.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))

        print("[INFO] loading model...")
        net = cv.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

        net.setInput(blob)
        start = time.time()
        preds = net.forward()
        end = time.time()
        print("[INFO] classification took {:.5} seconds".format(end - start))

        idxs = np.argsort(preds[0])[::-1][:5]

        for (i, idx) in enumerate(idxs):
            # draw the top prediction on the input image
            if resized.shape[0] == 180:
                resizedWinW = resizedWinH = winW * 4
                originX = x * 4
                originY = y * 4
            elif resized.shape[0] == 2 * 180:
                resizedWinW = resizedWinH = winW * 2
                originX = x * 2
                originY = y * 2
            else:
                resizedWinW = resizedWinH = winW
                originX = x
                originY = y

            if i == 0:
                if 'dog' in classes[idx] and (preds[0][idx] * 100) > 70:
                    cv.rectangle(imgCropped, (originX, originY), (originX + resizedWinW, originY + resizedWinH),
                                 (0, 255, 255), 2)
                    cv.putText(imgCropped, 'DOG', (originX + 5, originY + 25), cv.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 255), 2)
                elif 'cat' in classes[idx] and (preds[0][idx] * 100) > 90:
                    cv.rectangle(imgCropped, (originX, originY), (originX + resizedWinW, originY + resizedWinH),
                                 (0, 0, 255), 2)
                    cv.putText(imgCropped, 'CAT', (originX + 5, originY + 25), cv.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 0, 255), 2)

cv.imwrite("output.jpg", imgCropped)

cv.waitKey(0)
cv.destroyAllWindows()
