
import numpy as np
import cv2
from matplotlib import pyplot as plt

minLineLength = 2000
maxLineGap = 10

CAM_WIDTH = 1280
CAM_HEIGHT = 960

MIRROR = True

#marker_file_name = ["marker_one_small.png","marker_two_small.png","marker_three_small.png","marker_four_small.png"]
#marker_points = [[0,0],[0,background_height-100],[background_width-100,0],[background_width-100,background_height-100]]

def init_webcam(mirror=True):
    cam = []
    camera_height = []
    camera_width = []
    try:
        cam = cv2.VideoCapture(1)
        cam.set(cv2.CAP_PROP_FPS,50)
        cam.set(cv2.CAP_PROP_EXPOSURE,10)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_HEIGHT)
        ret_val, camera_image = cam.read()
        if len(camera_image) == 0:
            raise
        camera_height,camera_width,d = camera_image.shape
    except:
        print("No external camera found, attempting to default to  internal camera")
        cam = cv2.VideoCapture(0)
        ret_val, camera_image = cam.read()
        if len(camera_image) == 0:
            raise
        camera_height,camera_width,d = camera_image.shape
    return cam, camera_height, camera_width

def map_emoji_to_camera(corner_points, emoji_image):
    xs = [ p[0] for p in corner_points ]
    ys = [ p[1] for p in corner_points ]
    x_max = max(xs)
    x_min = min(xs)
    y_max = max(ys)
    y_min = min(ys)
    h,w,d = emoji_image.shape

    ordered_tracked = np.float32( [[x_min,y_min],[x_min,y_max],[x_max,y_max]] )
    ordered_emoji = np.float32([[0,0],[0,h-1],[w-1,h-1]])

    affine_mapping = cv2.getAffineTransform(ordered_emoji,ordered_tracked)
    dst = cv2.warpAffine(emoji_image,affine_mapping,(CAM_WIDTH,CAM_HEIGHT),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    return dst

def map_camera_to_projector(camera_image, affine_mapping):
    dst = cv2.warpAffine(camera_image,affine_mapping,(CAM_WIDTH,CAM_HEIGHT),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    return dst

def calculate_camera_affine_homography():
    affine_mapping = cv2.getAffineTransform(ordered_emoji, ordered_tracked)
    return affine_mapping

# Get camera image
cam, camera_height, camera_width = init_webcam()
print("Camera resolution", camera_width , camera_height)


smiley = cv2.imread('smiley.png')
plt.ion()

cv2.namedWindow("emoji", cv2.WINDOW_NORMAL) 
cv2.imshow('emoji', np.ones((CAM_WIDTH,CAM_HEIGHT),np.uint8))
cv2.namedWindow("hough", cv2.WINDOW_NORMAL) 
while True:
    ret_val, camera_image = cam.read()
    #if MIRROR:
    #    camera_image = cv2.flip( camera_image, 1 )

    camera_image = 255-camera_image
    gray = cv2.cvtColor(camera_image,cv2.COLOR_BGR2GRAY)
    low_threshold = np.max(gray) - 50
    edges = cv2.Canny(gray, low_threshold, low_threshold*3, apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    if lines is not None:
        points=[]
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                cv2.line(camera_image,(x1,y1),(x2,y2),(0,255,0),2)
                points.append([x1,y1])
                points.append([x2,y2])

        rect = cv2.minAreaRect(np.float32(points))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(camera_image,[box],0,(0,0,255),2)
    else:
        print("No lines found")

    cv2.imshow('hough', camera_image)
    try:
        cv2.imshow('emoji', map_emoji_to_camera(box, smiley) )
        print("Error")
    except:
        cv2.imshow('emoji', np.ones((CAM_WIDTH,CAM_HEIGHT),np.uint8))
        pass
    cv2.waitKey(30)