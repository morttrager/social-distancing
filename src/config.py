#Inference Details
SAVED_MODEL_PATH="models/efficientdet_d1_coco17_tpu-32/saved_model"
LABEL_MAP_PATH="label-maps/label_map.pbtxt"

# Input details
INPUT_VIDEO_PATH = "data/videos/vid_short.mp4"

#Output details
SAVE_VIDEO = False

# Check Perspective Transform - during this check inference is not done
CHECK_PERSPECTIVE = False   # to check the perspective transform of the Input Video
SAVE_FRAME = False          # to save the frame of the Input Video


#Application Config
APPLICATION_PORT = 5000

#Demo Only
CAM_DATA = {"1":INPUT_VIDEO_PATH}
CAM_DEPLOYMENT_STATUS = {}

