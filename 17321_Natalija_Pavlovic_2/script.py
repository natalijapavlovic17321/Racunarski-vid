import numpy as np
import cv2 as cv
MIN_MATCH_COUNT = 10

img1 = cv.imread('1.png')
img2 = cv.imread('2.png')
img3 = cv.imread('3.png')

detector = cv.xfeatures2d.SIFT_create()

kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches1 = flann.knnMatch(des2, des1, k=2)

good = []
for m, n in matches1:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.0)
    matchesMask1 = mask.ravel().tolist()
    result = cv.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[1]))
    result[0:img1.shape[0], 0:img1.shape[1]] = img1

    h, w, _ = result.shape
    result = result[0:h-100, 0:w - 185]

img = cv.drawMatches(img2, kp2, img1, kp1, good, None, matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask1, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
#cv.imshow("Match1", img)

kp4, des4 = detector.detectAndCompute(result, None)
kp3, des3 = detector.detectAndCompute(img3, None)
matches2 = flann.knnMatch(des3, des4, k=2)

for m, n in matches2:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp3[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp4[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask2 = mask.ravel().tolist()

    result2 = cv.warpPerspective(img3, M, (img3.shape[1] + result.shape[1], img3.shape[0] + 100))
    result2[0:result.shape[0], 0:result.shape[1]] = result

cv.imshow("Output2", result2)
cv.imwrite("output.jpg", result2)

img5 = cv.drawMatches(img3, kp3, img, kp4, good, None, matchColor=(0, 0, 255), singlePointColor=None, matchesMask=matchesMask2, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
#cv.imshow("Match2", img5)

cv.waitKey(0)