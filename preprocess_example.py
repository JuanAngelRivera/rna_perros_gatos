import numpy as np
import cv2


img = cv2.imread('example/original.jpg')

reescaled = cv2.resize(img, (128, 128))

cv2.imwrite('example/reescaled.jpg', reescaled)

gray = cv2.cvtColor(reescaled, cv2.COLOR_BGR2GRAY)

cv2.imwrite('example/gray.jpg', gray)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('example/binary.jpg', binary)

binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imwrite('example/binary_adaptive.jpg', binary_adaptive)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 100, 200)

cv2.imwrite('example/edges.jpg', edges)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_adaptive)

output = reescaled.copy()

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    print(area)

    if area >= 500:
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

if labels.max() > 0:
    label_img = np.uint8(255 * labels / labels.max())
else:
    label_img = labels

cv2.imwrite('example/labels.jpg', output)