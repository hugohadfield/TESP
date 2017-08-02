
import numpy as np
import cv2
import time

import math

minLineLength = 2000
maxLineGap = 10

CAM_WIDTH = 1280
CAM_HEIGHT = 960

BOX_SMOOTHING_FACTOR = 0.9

low_threshold_offset = 100

MIRROR = True

RANSAC_PARAMETER = 5.0

background_file_name = "square_white.png"

MIN_MATCH_COUNT = 6
MAX_MATCH_COUNT = 100

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1):
  return math.atan2( v1[1],v1[0] )


def compute_matches(background_des, camera_image, orb, bf):
    # find the keypoints and descriptors with ORB
    camera_kp = orb.detect(camera_image, None)
    matches = []
    if len(camera_kp) > 0:
        camera_kp, camera_des = orb.compute(camera_image, camera_kp)
        matches = bf.match(background_des, camera_des)
        if len(matches) > MAX_MATCH_COUNT:
            matches = sorted(matches, key = lambda x:x.distance)[0:MAX_MATCH_COUNT]
    return matches, camera_kp

def compute_homography(matches, background_kp, camera_kp):
    homography_mapping = []
    # Ensure we have enough matches to do homography calculation
    if len(matches)>MIN_MATCH_COUNT:
        # This was in the tutorial, will leave it be
        src_pts = np.float32([ background_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ camera_kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        # Do RANSAC
        homography_mapping, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,RANSAC_PARAMETER)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT))
        matchesMask = None
    return homography_mapping, matchesMask

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


def more_blue_than_red(color_in):
    dist_to_blue = (color_in[0] - 255)**2 + (color_in[2])**2
    dist_to_red = (color_in[0])**2 + (color_in[2] - 255)**2
    return dist_to_blue < dist_to_red


def test_for_good_lock(homography_mapping, background_height, background_width):
    pts = np.float32([ [0,0],[0,background_height-1],[background_width-1,background_height-1],[background_width-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,homography_mapping)
    area = cv2.contourArea(dst)
    print(area)
    if area > 1000:
        x,y,w,h = cv2.boundingRect(dst)
        metric = w*h
        error_ratio = abs(metric - area) / area
        if error_ratio < 0.8:
            return True
    else:
        return False


def show_matches(background_image, camera_image, background_kp, camera_kp, homography_mapping, matchesMask, matches):
    h,w,d = background_image.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,homography_mapping)
    camera_image = cv2.polylines(camera_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(background_image,background_kp,camera_image,camera_kp,matches,None,**draw_params)
    cv2.imshow('debug',img3)


def find_mapping_camera_to_projector(cam_object):
    # Load the marker image file 
    background_image = cv2.imread(background_file_name)
    h,w,d = background_image.shape
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #create BFMatcher object
    background_kp = orb.detect(background_image, None)
    background_kp, background_des = orb.compute(background_image, background_kp)

    lock = False
    while not lock:
        ret_val, camera_image = cam.read()
        matches, camera_kp = compute_matches(background_des, camera_image, orb, bf)
        if len(matches) <= MIN_MATCH_COUNT:
            cv2.waitKey(30)
            continue
        homography_mapping, matchesMask = compute_homography(matches, background_kp, camera_kp)
        show_matches(background_image, camera_image, background_kp, camera_kp, homography_mapping, matchesMask, matches)
        test_for_good_lock(homography_mapping, h, w)
        cv2.waitKey(30)

def get_bounding_box(camera_image):
    temp_image = 255-camera_image
    gray = cv2.cvtColor(temp_image,cv2.COLOR_BGR2GRAY)
    low_threshold = np.max(gray) - low_threshold_offset
    edges = cv2.Canny(gray, low_threshold, low_threshold*3, apertureSize = 3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    box = None
    if lines is not None:
        points=[]
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                #cv2.line(temp_image,(x1,y1),(x2,y2),(0,255,0),2)
                points.append([x1,y1])
                points.append([x2,y2])

        rect = cv2.minAreaRect(np.float32(points))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    return box


def test_for_good_box(box):
    if box is not None:
        area = cv2.contourArea(box)
        if area > 1000:
            x,y,w,h = cv2.boundingRect(box)
            metric = w*h
            error_ratio = abs(metric - area) / area
            if error_ratio < 0.8:
                return True
        else:
            return False
    else:
        return False

def subtract_points(p1,p2):
    return [p1[0]-p2[0],p1[1]-p2[1]]

def sort_box(box_unsorted):
    xs = [p[0] for p in box_unsorted]
    ys = [p[1] for p in box_unsorted]
    centroid = [sum(xs)/len(xs), sum(ys)/len(ys)]
    def angle_to_centroid(box_point):
        return angle(subtract_points(box_point, centroid))
    return sorted(box_unsorted, key=angle_to_centroid)


def smooth_box(BOX_SMOOTHING_FACTOR,box_old,box_unsorted):
    box_new = sort_box(box_unsorted)
    box_temp = []
    for p_index in range(len(box_old)):
            box_temp.append([box_old[p_index][0]*BOX_SMOOTHING_FACTOR + box_new[p_index][0]*(1-BOX_SMOOTHING_FACTOR) , \
            box_old[p_index][1]*BOX_SMOOTHING_FACTOR + box_new[p_index][1]*(1-BOX_SMOOTHING_FACTOR) ])
    return box_temp

def find_stable_box_mask(duration):
    t = time.time()
    lock_aquired = False
    box_old = np.float32([[0,0],[0,0],[0,0],[0,0]])
    while True:
        if not lock_aquired:
            t = time.time()
        ret_val, camera_image = cam.read()
        
        box = get_bounding_box(camera_image)
        lock_aquired = test_for_good_box(box)
        if lock_aquired:
            box_old = smooth_box(BOX_SMOOTHING_FACTOR,box_old,box)
            cv2.drawContours(camera_image,[box],0,(0,0,255),2)
            cv2.drawContours(camera_image,np.int0([box_old]),0,(0,255,0),2)
        cv2.imshow('debug',camera_image)
        cv2.waitKey(30)
        elapsed = time.time() - t
        if elapsed > duration:
            return box_old



# Get camera image
cam, camera_height, camera_width = init_webcam()
print("Camera resolution", camera_width , camera_height)

smiley = cv2.imread('smiley.png')

cv2.namedWindow("emoji", cv2.WINDOW_NORMAL) 
cv2.namedWindow("debug", cv2.WINDOW_NORMAL) 
cv2.imshow('emoji', 255*np.ones((CAM_WIDTH,CAM_HEIGHT,3),np.uint8))



# Find the box
find_stable_box_mask(100)
# Use the box to make a mask

# Use the mask to find the homography

cv2.namedWindow("hough", cv2.WINDOW_NORMAL) 
#find_mapping_camera_to_projector(cam)
while True:
    ret_val, camera_image = cam.read()
    #if MIRROR:
    #    camera_image = cv2.flip( camera_image, 1 )
    box = get_bounding_box(camera_image)
    if box is not None:
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