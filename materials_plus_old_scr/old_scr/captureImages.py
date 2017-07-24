# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:43:04 2016

@author: skimmer

automatically finding matching points, calculate H, process images to compare, plots
"""
import cv2 
import numpy as np

# Settings
imgSaveDir = "./imgs5"
imgFileNameDesc = "img%03d.png"
imgPathToImageToDisplay = "./layerImgs/original.png"
numImgsToCapture = 200
pauseBetweenCaptures = 2000 #[mseconds]

cameraDeviceIndex = 1 #choose camera device [0,N-1], 0 for first device, 1 for second device etc.

# GO

cap = cv2.VideoCapture(cameraDeviceIndex)

# destroy old windows (if they exists)
cv2.destroyAllWindows()

# Load and display image
imgToDisplay = cv2.imread(imgPathToImageToDisplay, 0)
cv2.imshow("background", imgToDisplay)


# Make photos with webcam
numImgCaptured = 0
while numImgCaptured < numImgsToCapture:
    
    ret, img = cap.read()
    if ret == False:
        print("Img capture failed")
        break
      
        
    cv2.imshow("captured", img)
    imgFileName = imgFileNameDesc % (numImgCaptured)
    imgFilePath = imgSaveDir + "/" + imgFileName
    cv2.imwrite(imgFilePath, img)
    
    numImgCaptured = numImgCaptured + 1
    #key = cv2.waitKey(pauseBetweenCaptures)
    key = cv2.waitKey(2)
    if key == 27:
        cv2.destroyAllWindows()
        break

cap.release()