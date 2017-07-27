
import numpy as np
import cv2

######### Parameters #########
MIN_MATCH_COUNT = 10


######### Function Definitions #########

def init_webcam(mirror=False):
    cam = []
    try:
        cam = cv2.VideoCapture(1)
        ret_val, camera_image = cam.read()
        if len(camera_image) == 0:
            raise
    except:
        print("No external camera found, attempting to default to  internal camera")
        cam = cv2.VideoCapture(0)
    return cam

def show_webcam(mirror=False):
    cam = []
    try:
        cam = cv2.VideoCapture(1)
        ret_val, camera_image = cam.read()
        if len(camera_image) == 0:
            raise
    except:
        print("No external camera found, attempting to default to  internal camera")
        cam = cv2.VideoCapture(0)
    while True:
        ret_val, camera_image = cam.read()
        if mirror: 
            camera_image = cv2.flip(camera_image, 1)
        compute_matches(reference_image,camera_image)
    cv2.destroyAllWindows()

def show_matches(background_image, camera_image, background_kp, camera_kp, homography_mapping):
    h,w,d = background_image.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,homography_mapping)
    camera_image = cv2.polylines(camera_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(background_image,background_kp,camera_image,camera_kp,matches,None,**draw_params)
    cv2.imshow('Projector',img3)
    cv2.waitKey(30)

def compute_matches(background_des, camera_image):
    # find the keypoints and descriptors with ORB
    camera_kp = orb.detect(camera_image, None)
    camera_kp, camera_des = orb.compute(camera_image, camera_kp)
    # Match the descriptors between images
    matches = bf.match(background_des, camera_des)
    return matches, camera_kp

def compute_homography(matches, background_kp, camera_kp):
    homography_mapping = []
    # Ensure we have enough matches to do homography calculation
    if len(matches)>MIN_MATCH_COUNT:
        # This was in the tutorial, will leave it be
        src_pts = np.float32([ background_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ camera_kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        # Do RANSAC
        homography_mapping, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT))
        matchesMask = None
    return homography_mapping, matchesMask

def magnify_area(center_point, magnification_scale, rotation_angle_degrees):
    pass

if __name__ == '__main__':

    ######### Initialisation #########

    # Set up the camera
    cam = init_webcam()

    # Init ORB detector and feature matcher
    cv2.ocl.setUseOpenCL(False) #bugfix
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #create BFMatcher object

    # Load the background image and compute markers on it
    background_image = cv2.imread('background.jpg')
    background_kp = orb.detect(background_image, None)
    background_kp, background_des = orb.compute(background_image, background_kp)

    # Create an opencv window to display the projection into
    cv2.namedWindow("Projector", cv2.WINDOW_NORMAL) 

    ######### Main Loop #########
    while True:
        # Get an image from the camera
        ret_val, camera_image = cam.read()

        # Detect the homography between the background image and the camera
        matches, camera_kp = compute_matches(background_des, camera_image)
        homography_mapping, matchesMask = compute_homography(matches, background_kp, camera_kp)
        show_matches(background_image, camera_image, background_kp, camera_kp, homography_mapping)

        # Define a point about which we will magnify the image and the 
        # Cut that area out from the original image, magnify it and crop it
        magnify_area(center_point, magnification_scale, rotation_angle_degrees)

        # Finish up and project it!
