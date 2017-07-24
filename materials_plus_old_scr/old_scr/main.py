# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:43:04 2016

@author: ROMANMUELLER

automatically finding matching points, calculate H, process images to compare, plots
"""
import cv2
import numpy as np

######################
# SETTIGNS

option = 1 #choose 0 for files, 1 for webcam

#images
imgLayer0Path = "./layerImgs/skin.png"
imgLayer1Path = "./layerImgs/muscles.png"
imgLayer2Path = "./layerImgs/skeleton.png"
imgLayer3Path = "./layerImgs/skeleton.png"
imgLayer4Path = "./layerImgs/skeleton.png"

imgMaskPath = "./layerImgs/maskRect_real.png"

cameraDeviceIndex = 1 #choose camera device [0,N-1], 0 for first device, 1 for second device etc.

# For captured camera images:
imgSourceDir = "./imgs2"
imgFileNameDesc = "img%03d.png"
imgStartIndex = 0
imgEndIndex = 200

displayOnProjector = True
determineKeyPointsFromMergedImg = True

### Parameters
minimumMatchCount = 8
homographyThreshold = 10
minKeypoints = 10
# For H feasability check
margin = (1800, 1600) #(x,y) x=>width, y=>height


#########################
# Functions
  
    
#rotation detection
# if no rotation, Rot = 0; if anticlockwise rotation(90+-30), Rot = 1, if 
#clockwise rotation(90+-30), Rot = 2; else Rot = 3 

def proj(H, p):
    ph = np.array([p[0], p[1], 1])
    pw = np.dot(H, ph)
    pt = np.array([pw[0] / pw[2], pw[1] / pw[2]])
    return pt
    
def checkH(H, size, margin):
    # size = (width, height)
    width = size[0] - 1
    height = size[1] - 1
    
    #  ----> x (width)
    # |
    # |  p0m ---------- p1m
    # v
    # y     p0 --- p1
    #       |      |
    #       p3 --- p2
    #
    #    p3m ---------- p2m
    
    p0 = np.array([0, 0])    
    p1 = np.array([width, 0])
    p2 = np.array([width, height])
    p3 = np.array([0, height])

    p0m = p0 - margin   
    p2m = p2 + margin
    
    # Projected points
    p0w = proj(H, p0)
    p1w = proj(H, p1)
    p2w = proj(H, p2)
    p3w = proj(H, p3)
    
    isInside = False  
    isConvex = False  
    
    # Check if point projections are inside margin
    if all(p0w > p0m) & all(p0w < p2m) & \
        all(p1w > p0m) & all(p1w < p2m) & \
        all(p2w > p0m) & all(p2w < p2m) & \
        all(p3w > p0m) & all(p3w < p2m):
        isInside = True    
    #else:
        #print("is NOT Inside")
    
    # Check if points are convex
    cp0 = np.cross(p1w - p0w, p2w - p1w)
    cp1 = np.cross(p2w - p1w, p3w - p2w)
    cp2 = np.cross(p3w - p2w, p0w - p3w)
    cp3 = np.cross(p0w - p3w, p1w - p0w)
    if (cp0 > 0) & (cp1 > 0) & (cp2 > 0) & (cp3 > 0):
        isConvex = True
    #else:
        #print("is NOT Convex")
        
    isFeasable = isInside & isConvex
    
    return isFeasable  



def rotation(H, RotIndOrig):        #Rotation dection
    RotIndProj = np.dot(H, RotIndOrig)
    RotDis = RotIndProj[1,0] - RotIndProj[1,1]
    if np.abs(RotDis) < 240:#240 = 480*sin30
        if RotIndProj[0,0] < RotIndProj[0,1]:
            Rot = 1 #anticlockwise rotation
        else:
            Rot = 2 #clockwise rotation
    elif np.abs(RotDis) > 339:#339 = 480*cos30
        Rot = 0
    else:
        Rot = 3
#    print Rot
    return Rot
    
    
# mode switch
def switch(Rot, counter, numOfH, mode):
    if Rot == 1:
        counter = 3
    elif counter > 0 :
        counter = counter - 1
                
    if counter > 0:
        numOfH = numOfH + 1
    else:
        numOfH = 0
       
    #print "Rot = %d" % Rot
    #print "numOfH = %d"%numOfH        
    
    if numOfH >= 15:        
        if mode >= 3:# 3 modes
            mode = 0
        else:
            mode = mode + 1
        numOfH = 0
        counter = 0            

    #print "mode = %d\n" % mode
    return (mode,counter,numOfH)        


def overlayWarp(H, backgroundImg, overlayImg, mask):
    
#    maskSize = (imgMask.shape[1], imgMask.shape[0])
#    maskCenter = np.array([maskSize[0] / 2, maskSize[1] / 2, 1])    
        
    imgSize = (backgroundImg.shape[1], backgroundImg.shape[0])
    overlayImgMasked = overlayImg * mask
    imgProjected = cv2.warpPerspective(overlayImgMasked, H, imgSize)

    validIndex = imgProjected != 0
    
    overlaidImg = backgroundImg.copy()

    overlaidImg[validIndex] = overlayImg[validIndex]
    
    return overlaidImg

#########################
# Do IT


imgLayer0 = cv2.imread(imgLayer0Path, 1)
imgLayer1 = cv2.imread(imgLayer1Path, 1)
imgLayer2 = cv2.imread(imgLayer2Path, 1)
imgLayer3 = cv2.imread(imgLayer3Path, 1)
imgLayer4 = cv2.imread(imgLayer4Path, 1)

### Load an prepare mask
imgMaskOrig = cv2.imread(imgMaskPath, 1)
if imgMaskOrig is None:
    print("The mas image " + imgMaskPath + " could not be read")

threshhold = 125
imgMask = imgMaskOrig.copy()
imgMask[imgMaskOrig >= threshhold] = 0
imgMask[imgMaskOrig < threshhold] = 1


maskSize = (imgMask.shape[1], imgMask.shape[0])
maskCenter = np.array([maskSize[0] / 2, maskSize[1] / 2, 1])



#Init ORB detector and feature matcher
cv2.ocl.setUseOpenCL(False) #bugfix
orb = cv2.ORB_create()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #create BFMatcher object

keyPoints = orb.detect(imgLayer0, None)
keyPoints, desPoints = orb.compute(imgLayer0, keyPoints)

if option == 1:
    #cam initialzation
    cap = cv2.VideoCapture(cameraDeviceIndex)
    
    if cap.isOpened(): # try to get the first frame
        print("Opened camera stream!")
        ret, frame = cap.read()
        if ret == False:
            print("Test capture failed!")
            quit()
    else:
        print("Open capture failed!!!")
else:
    # Read images
    frame = 0


# Create Window for fullscreen display
if displayOnProjector == True:
    cv2.namedWindow("projector", cv2.WND_PROP_FULLSCREEN) 
    cv2.namedWindow("projector") 
    cv2.moveWindow("projector", 1920, 0)         
    cv2.setWindowProperty("projector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#cv2.setWindowProperty("projector", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
#cv2.setWindowProperty("projector", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)

cv2.imshow("projector", imgLayer0)    
imgSize = (imgLayer0.shape[1], imgLayer0.shape[0]) #(width, height)
cv2.waitKey(1)
    
#Loop initialization
imgProj = imgLayer0
imgIndex = imgStartIndex
doLoop = True
# init out image
imgMatchesVisu = 0

RotIndOrig = np.array([[320,320],[0,480],[1,1]])
Rot = 0
mode = 0 #mode
numOfH = 0 #number of 'high' states
counter = 0 #noise toleration 


while doLoop:
 
    
    if option == 1:
        # Web cam
        ret, capturedImg = cap.read()
        if ret == False:
            print("Img capture failed")
            break
    else:
        if imgIndex > imgEndIndex:
            break
        # File Source
        imgFileName = imgFileNameDesc % (imgIndex)
        imgFilePath = imgSourceDir + "/" + imgFileName
        print("Loading: " + imgFilePath)
        capturedImg = cv2.imread(imgFilePath, 1)
        imgIndex = imgIndex + 1
        
    
    # Find features in projector img
    if determineKeyPointsFromMergedImg == True:
        kpProjImg = orb.detect(imgProj, None) 
        kpProjImg, desProjImg = orb.compute(imgProj, kpProjImg)    
        keyPoints = kpProjImg
        desPoints = desProjImg
    
    # Find features in curent captured img
    kpCapturedImg = orb.detect(capturedImg, None) 
    kpCapturedImg, desCapturedImg = orb.compute(capturedImg, kpCapturedImg)
    

    if len(kpCapturedImg) > minKeypoints:
   
        #find matching points
        matches = bf.match(desPoints, desCapturedImg) # Match descriptors
            
        if len(matches) > minimumMatchCount:
            #print(len(matches))
            ptsOriginal = np.float32([ keyPoints[m.queryIdx].pt for m in matches ])
            ptsCaptured = np.float32([ kpCapturedImg[m.trainIdx].pt for m in matches ])
    
            if (len(keyPoints) > minKeypoints) & (len(kpCapturedImg) > minKeypoints) \
                & (len(matches) > minimumMatchCount)\
                & (len(keyPoints) >= len(kpCapturedImg)):
                # Note that  matchesMask from 
                # cv2.drawMatches() does not seem to work with the mask
                
                if determineKeyPointsFromMergedImg == True:
                    imgMatchesVisu = cv2.drawMatches(imgProj, keyPoints, capturedImg, kpCapturedImg, \
                        matches, imgMatchesVisu)
                else:
                    imgMatchesVisu = cv2.drawMatches(capturedImg, kpCapturedImg, \
                        imgLayer0, keyPoints, matches, imgMatchesVisu)
                        
                
                cv2.imshow("Matches", imgMatchesVisu)
        
        
            H, mask = cv2.findHomography(ptsCaptured, ptsOriginal, cv2.RANSAC, homographyThreshold)
            if checkH(H, imgSize, margin) == True:
                # Rotation detection
                #Rot = rotation(H, RotIndOrig)
                #mode,counter,numOfH = switch(Rot, counter, numOfH, mode)
                
                #flashlight Overlay
                imgMaskProj = cv2.warpPerspective(imgMask, H, maskSize)
                
                
                if mode == 0:    
                    currentLayerImg = imgLayer1        
                elif mode == 1:
                    currentLayerImg = imgLayer2
                elif mode == 2:
                    currentLayerImg = imgLayer3
                else:        
                    currentLayerImg = imgLayer4
            
                imgProj = np.multiply(currentLayerImg, imgMaskProj) + np.multiply(imgLayer0, (1-imgMaskProj))
                
                imgProj = overlayWarp(H, imgLayer0, currentLayerImg, imgMask)
 
            else:
                print("H is NOT feasable")
        else:
            print("To few matching points")
    else:
        print("Not enough keypoints")

    # Display stuff
    cv2.imshow("projector", imgProj)
    # cv2.imshow("webcam", capturedImg)
    

    #cancelation criteria
    # note: waitkey does the event procesing of e.g. the imshow function
    key = cv2.waitKey(2)
    if (key == 27) | (imgIndex == imgEndIndex):
        cv2.destroyAllWindows()
        
        if option == 1:
            print("Releasing image capture device")
            cap.release
            
        doLoop = False
        break
    
