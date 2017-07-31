
import numpy as np
import cv2
from numpy.linalg import inv

######### Parameters #########
MIN_MATCH_COUNT = 6
MAX_MATCH_COUNT = 11
EXP_SMOOTHING_FACTOR = 0.5
MATRIX_EXP_SMOOTHING_FACTOR = 0.5
RANSAC_PARAMETER = 5.0
#background_height = 1126
#background_width = 1772
#background_height = 935 
#background_width = 1296
background_height = 981 
background_width = 1590
#background_height = 623 
#background_width = 1146
delta_t = 1
background_file_name = 'wheresWally3.jpg'
magnifying_file_name = 'magnifying_glass.png'

SIFT_FLAG = False
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)
search_params = dict(checks = 50)

####### Globals #######
CAM_WIDTH = 0
CAM_HEIGHT = 0
tracked_centre_point = [0,0]
tracked_velocity = [0,0]
smoothed_mapping = np.float32([[1, 0, 0],[0,1,0],[0,0,1]])

######### Function Definitions #########
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
    cv2.imshow('Debug',img3)

def compute_matches(background_des, camera_image):
    # find the keypoints and descriptors with ORB
    if SIFT_FLAG: 
        camera_kp = sift.detect(camera_image, None)
    else:
        camera_kp = orb.detect(camera_image, None)
    matches = []
    if len(camera_kp) > 0:
        if SIFT_FLAG: 
            camera_kp, camera_des = sift.compute(camera_image, camera_kp)
            temp_matches  = flann.knnMatch(background_des,camera_des,k=2)
            for m,n in temp_matches:
                if m.distance < 0.7*n.distance:
                    matches.append(m)
        else:
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

def magnify_area(background_image, center_point, magnification_scale, rotation_angle_degrees):
    # Build a binary mask of the area of interest
    circle_mask = np.zeros((background_height,background_width), np.uint8)
    cv2.circle(circle_mask,tuple(center_point),200,1,thickness=-1)
    # Use the mask to cut out the area that we need
    roi_image = cv2.bitwise_and(background_image, background_image, mask=circle_mask)
    roi_image = cv2.resize(roi_image,None,fx=magnification_scale, fy=magnification_scale, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('Debug2',roi_image)
    new_centre = [magnification_scale*p for p in center_point]
    translation_vector = [0,0]
    for i in range(2):
        translation_vector[i] = center_point[i] - new_centre[i]
    M = np.float32([[1,0,translation_vector[0]],[0,1,translation_vector[1]]])
    rows,cols,d = roi_image.shape
    dst = cv2.warpAffine(roi_image,M,(cols,rows))
    moved_and_cropped = dst[0:background_height, 0:background_width, :].copy()
    moved_and_cropped = cv2.bitwise_and(moved_and_cropped, moved_and_cropped, mask=circle_mask)

    # Invert the mask to show the area that is kept
    circle_mask = 1-circle_mask
    compound_image = cv2.bitwise_and(background_image, background_image, mask=circle_mask)
    compound_image = cv2.bitwise_or(compound_image, moved_and_cropped)
    cv2.circle(compound_image,tuple(center_point),200,1,thickness=15)
    return compound_image

def get_central_camera_point(homography_mapping):
    src = np.float32([ [round(CAM_WIDTH/2),round(CAM_HEIGHT/2)] ]).reshape(-1,1,2)
    m = cv2.invert(homography_mapping) 
    dst = cv2.perspectiveTransform( src, m[1])
    return dst

def test_for_good_lock(homography_mapping, background_height, background_width):
    pts = np.float32([ [0,0],[0,background_height-1],[background_width-1,background_height-1],[background_width-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,homography_mapping)
    area = cv2.contourArea(dst)
    if area > 100000:
        x,y,w,h = cv2.boundingRect(dst)
        metric = w*h
        error_ratio = abs(metric - area) / area
        if error_ratio < 0.8:
            return True
    else:
        return False

def smooth_matrix(homography_mapping):
    for j in range(3):
        for i in range(3):
            smoothed_mapping[i][j] = smoothed_mapping[i][j]*(MATRIX_EXP_SMOOTHING_FACTOR) + homography_mapping[i][j]*(1-MATRIX_EXP_SMOOTHING_FACTOR)


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def smooth_centre_motion(measured_position, delta_t):
    new_point = [0,0]
    for i in range(2):
        new_point[i] = int(round((tracked_centre_point[i] + tracked_velocity[i]*delta_t)*MATRIX_EXP_SMOOTHING_FACTOR + measured_position[i]*(1-MATRIX_EXP_SMOOTHING_FACTOR)))
        tracked_velocity[i] = new_point[i] - tracked_centre_point[i]
        tracked_centre_point[i] = new_point[i]


if __name__ == '__main__':

    ######### Initialisation #########

    # Set up the camera
    cam, camera_height, camera_width = init_webcam()
    CAM_WIDTH = camera_width
    CAM_HEIGHT = camera_height

    # Load the background image and compute markers on it
    background_image = cv2.imread(background_file_name)
    h,w,d = background_image.shape

    # Init ORB detector and feature matcher
    cv2.ocl.setUseOpenCL(False) #bugfix
    if SIFT_FLAG:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)
        background_sift = cv2.xfeatures2d.SIFT_create(nfeatures=2000)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        print("About to compute")
        temp_kp = background_sift.detect(background_image, None)
        temp_kp, temp_des = background_sift.compute(background_image, temp_kp)
        background_des = temp_des#[0:10000]
        background_kp = temp_kp#[0:10000]
        print("Computed features")
    else:
        orb = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #create BFMatcher object

    # Create an opencv window to display the projection into
    cv2.namedWindow("Projector", cv2.WINDOW_NORMAL) 
    cv2.imshow('Projector',background_image)
    cv2.namedWindow("Debug", cv2.WINDOW_NORMAL) 
    #cv2.namedWindow("Debug2", cv2.WINDOW_NORMAL) 

    compound_image = background_image.copy()

    ######### Main Loop #########
    while True:
        if not SIFT_FLAG:
            orb2 = cv2.ORB_create(nfeatures=500)
            background_kp = orb2.detect(compound_image, None)
            background_kp, background_des = orb2.compute(compound_image, background_kp)

        # Get an image from the camera
        ret_val, camera_image = cam.read()

        # Detect the homography between the background image and the camera
        matches, camera_kp = compute_matches(background_des, camera_image)
        if len(matches) <= MIN_MATCH_COUNT:
            cv2.waitKey(30)
            print("FAIL matches")
            continue
        homography_mapping, matchesMask = compute_homography(matches, background_kp, camera_kp)
        show_matches(compound_image, camera_image, background_kp, camera_kp, homography_mapping)

        # If it is square update our position to the new found position, else keep the old one
        if test_for_good_lock(homography_mapping, background_width, background_height):
            # Define a point about which we will magnify the image and the rotation angle
            smooth_matrix(homography_mapping)
            dst = get_central_camera_point(smoothed_mapping)
            new_point = dst[0][0]
        else:
            new_point = [p for p in tracked_centre_point]
            print("FAIL lock")
        smooth_centre_motion(new_point, delta_t)

        # Cut that area out from the original image, magnify it and crop it
        magnification_scale = 2
        rotation_angle_degrees = []
        compound_image = magnify_area(background_image, tracked_centre_point, magnification_scale, rotation_angle_degrees)

        # Finish up and project it!
        cv2.imshow('Projector',compound_image)
        cv2.waitKey(30)
