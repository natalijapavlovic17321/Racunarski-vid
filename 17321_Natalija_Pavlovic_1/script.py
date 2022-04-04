import numpy as np
import cv2 as cv

imgInput = cv.imread('input.png')
cv.imshow('Input', imgInput)

imgMedian = cv.medianBlur(imgInput, 11)
cv.imshow('medianBlur', imgMedian)

imgHSV = cv.cvtColor(imgMedian, cv.COLOR_BGR2HSV)
#cv.imshow("Saturation", imgHSV[:, :, 1])
#cv.imshow("Value", imgHSV[:, :, 2])
imgHue = imgHSV[:, :, 0]
cv.imshow("Hue", imgHue)

imgTreshold = cv.inRange(imgHue, 20, 100)
cv.imshow("Treshold", imgTreshold)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(11, 11))
imgOpen = cv.erode(imgTreshold, kernel=kernel)
imgOpen = cv.dilate(imgOpen, kernel=kernel)
cv.imshow("Open", imgOpen)

cntCC, imgCC = cv.connectedComponents(imgOpen, connectivity=4)
maxCnt = 0
maxBBox = None
imgOut = imgInput.copy()
for cc in range(1, cntCC):
    imgCurr = np.where(imgCC == cc, 255, 0).astype(np.uint8)
    x, y, w, h = cv.boundingRect(imgCurr)
    cnt = imgCurr.sum()/255
    if cnt > maxCnt:
        maxCnt = cnt
        maxBBox = x, y, w, h
    cv.rectangle(imgOut, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)

x, y, w, h = maxBBox
cv.rectangle(imgOut, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
imgMaxCC = imgOut[y:y+h, x:x+w, :]
cv.imshow("MaxCC", imgMaxCC)

cv.putText(imgOut, text='Count:' + str(cntCC), org=(5, 17), fontFace=cv.FONT_HERSHEY_SIMPLEX,
           fontScale=0.5, color=(0, 0, 255), thickness=2)
cv.imshow("Output", imgOut)

cv.imwrite("input.png", imgInput)
cv.imwrite("range.png", imgTreshold)
cv.imwrite("output.png", imgOut)

cv.waitKey(0)
cv.destroyAllWindows()