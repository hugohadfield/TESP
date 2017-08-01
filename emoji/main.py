
import numpy as np
import cv2

# Total resolution
total_width = 1200
total_height = 700

def init_webcam(mirror=False):
    cam = []
    camera_height = []
    camera_width = []
    try:
        cam = cv2.VideoCapture(1)
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

def draw_emoji(corner_points, emoji_image):
	xs = [ p[0] for p in corner_points ]
	ys = [ p[1] for p in corner_points ]
	x_max = max(xs)
	x_min = min(xs)
	y_max = max(ys)
	y_min = min(ys)
	h,w,d = emoji_image.shape
	x_scale = (x_max - x_min)/w
	y_scale = (y_max - y_min)/h
	scaled_emoji = cv2.resize(emoji_image,None, fx=x_scale, fy=y_scale, interpolation = cv2.INTER_CUBIC)
	blank_image = np.zeros((total_height,total_width,3), np.uint8)
	blank_image[y_min:y_max,x_min:x_max] = scaled_emoji.copy()
	return blank_image

# Get camera image
cam, camera_height, camera_width = init_webcam()
print("Camera resolution", camera_width , camera_height)
ret_val, camera_image = cam.read()

smiley = cv2.imread('smiley.png')
gray = cv2.cvtColor(camera_image,cv2.COLOR_BGR2GRAY)
gray_invert = 255 - gray

minLineLength = 50
maxLineGap = 30
lines = cv2.HoughLinesP(gray_invert,1,np.pi/180,15,minLineLength,maxLineGap)
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

cv2.imshow('emoji',draw_emoji(box, smiley) )

cv2.imshow('hough', camera_image)
cv2.waitKey(0)