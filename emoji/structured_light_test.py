
import numpy as np
import cv2

minLineLength = 2000
maxLineGap = 10

CAM_WIDTH = 1280
CAM_HEIGHT = 960

low_threshold_offset = 100

MIRROR = True


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


def more_blue_than_red(color_in):
    dist_to_blue = (color_in[0] - 255)**2 + (color_in[2])**2
    dist_to_red = (color_in[0])**2 + (color_in[2] - 255)**2
    return dist_to_blue < dist_to_red

def generate_structured_pattern(pattern_level):
    blue =  (255,0,0)
    red = (0,0,255)
    blank_image = np.zeros((CAM_HEIGHT,CAM_WIDTH,3), np.uint8)
    if pattern_level == 0:
        blank_image[:,0:int(0.5*CAM_WIDTH)] = blue     # (B, G, R)
        blank_image[:,int(0.5*CAM_WIDTH):CAM_WIDTH] = red
    elif pattern_level == 1:
        blank_image[0:int(0.5*CAM_HEIGHT),:] = blue     # (B, G, R)
        blank_image[int(0.5*CAM_HEIGHT):CAM_HEIGHT,:] = red
    elif pattern_level == 2:
        blank_image[0:int(0.5*CAM_HEIGHT),:] = blue     # (B, G, R)
        blank_image[int(0.5*CAM_HEIGHT):CAM_HEIGHT,:] = red
    return blank_image

def find_mapping_camera_to_projector(cam_object, window_name, corner_points_camera_coords):
    level_list = range(2)
    position_code = []

    for level in level_list:
        # Project structured light
        patterned_image = generate_structured_pattern(level)
        cv2.imshow(window_name, patterned_image)
        cv2.waitKey(50)

        # Get n frames at the new color
        center_point = [int(CAM_WIDTH/3),int(CAM_HEIGHT/2)]
        sum_blue = 0
        n = 4
        ret_val, camera_image = cam.read()
        for i in range(n):
            ret_val, camera_image = cam.read()
            temp_im = camera_image.copy()
            sum_blue += more_blue_than_red(camera_image[center_point[0],center_point[1],:]) 
            cv2.waitKey(10)
        temp_im = camera_image.copy()
        cv2.circle(temp_im,tuple(center_point),20,1,thickness=1)
        cv2.imshow("debug", temp_im)
        position_code.append(sum_blue > (float(n)/2))
    print(position_code)




cam, camera_height, camera_width = init_webcam()
print("Camera resolution", camera_width , camera_height)

cv2.namedWindow("emoji", cv2.WINDOW_NORMAL) 
while True:
	corner_points_camera_coords = [[100,100]]
	find_mapping_camera_to_projector(cam, "emoji", corner_points_camera_coords)