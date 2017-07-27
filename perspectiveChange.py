import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('wheresWally.jpg')
# rows,cols,ch = img.shape
img2 = img1[200:400, 100:300]

rows, cols = img2.shape[:2]
M = cv2.getRotationMatrix2D((cols/2,rows/2),20,1)

img3 = cv2.warpAffine(img2,M,(cols,rows))


# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

# M = cv2.getPerspectiveTransform(pts1,pts2)

# print(img.shape) 
# dst = cv2.warpPerspective(img,M,(rows,cols))

plt.subplot(131),plt.imshow(img1),plt.title('Input')
plt.subplot(132),plt.imshow(img2),plt.title('Output')
plt.subplot(133),plt.imshow(img3),plt.title('   ')
plt.show()