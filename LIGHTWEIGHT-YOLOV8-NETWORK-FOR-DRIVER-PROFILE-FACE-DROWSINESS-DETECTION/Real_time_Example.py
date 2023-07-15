import cv2
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
#%matplotlib inline
import ultralytics
import onnxruntime
from ultralytics import YOLO
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils import ops
import torch
import math
import datetime
# GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# model
model = YOLO(r'C:\Users\zhang\Desktop\PDMS\output (1)\work\pdms\pdms0\comparision\onnx\fastnet.onnx', task='pose')
#model.to(device)

# rectangle
bbox_color = (150, 0, 0)             
bbox_thickness = 2                   

# rectangle comment
bbox_labelstr = {
    'font_size':1,         
    'font_thickness':2,    
    'offset_x':0,          
    'offset_y':-10,        
}

# key points BGR
kpt_color_map = {
    0:{'name':'1', 'color':[255, 0, 0], 'radius':6},      # Outer edge of tragus
    1:{'name':'2', 'color':[0, 255, 0], 'radius':6},      # Outer eye corner
    2:{'name':'3', 'color':[0, 0, 255], 'radius':6},      # Tip of the nose
    3:{'name':'4', 'color':[255, 0, 255], 'radius':6},      # Superior lip margin
    4:{'name':'5', 'color':[0, 255, 255], 'radius':6},      # Outer corners of the mouth
}

# keypoint comment
kpt_labelstr = {
    'font_size':1.5,             
    'font_thickness':3,       
    'offset_x':10,             
    'offset_y':0,            
}
alpha_string='null'
beta_string='null'


def process_frame(img_bgr):

    
    # record the starting time for each frame
    start_time = time.time()
    
    results = model(img_bgr, verbose=False) 
    
    # how many targets
    num_bbox = len(results[0].boxes.cls)
    
    # rectangle xyxy
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32') 
    
    print(results[0].keypoints)
    # keypoint xy
    bboxes_keypoints = results[0].keypoints.xy.cpu().numpy().astype('uint32')
    
    print(bboxes_keypoints)
    
    for idx in range(num_bbox): # all rectangle

        # rectangle xyxy
        bbox_xyxy = bboxes_xyxy[idx] 

        # rectangle type
        bbox_label = results[0].names[0]

        # draw rectangle
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness)

        # comment of rectangle
        img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

        bbox_keypoints = bboxes_keypoints[idx] # all keypoints xy and confidence

   
        # draw key points
        for kpt_id in kpt_color_map:

            # color, radius, xy
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]

            # draw point
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
            
            x_string = 'x  {:.2f}'.format(kpt_x) 
            img_bgr = cv2.putText(img_bgr, x_string, (kpt_x, kpt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            y_string = 'y  {:.2f}'.format(kpt_y) 
            img_bgr = cv2.putText(img_bgr, y_string, (kpt_x, kpt_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            

            # keypoints comments
            kpt_label = str(kpt_id) # keypoints id
            # kpt_label = str(kpt_color_map[kpt_id]['name'])
            img_bgr = cv2.putText(img_bgr, kpt_label, (kpt_x+kpt_labelstr['offset_x'], kpt_y+kpt_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color, kpt_labelstr['font_thickness'])
     
        #line 1
        
        global alpha_string
        global beta_string
        
        ptStart = (bbox_keypoints[0][0], bbox_keypoints[0][1])
        ptEnd = (bbox_keypoints[2][0], bbox_keypoints[2][1])
        point_color = (0, 255, 0) # BGR
        thickness = 2 
        lineType = 4
        cv2.line(img_bgr, ptStart, ptEnd, point_color, thickness, lineType)
        #arctan1=abs(bbox_keypoints[0][1]-bbox_keypoints[2][1])/abs(bbox_keypoints[2][0]-bbox_keypoints[0][0])
        #这个地方有问题，用if else解决
        if bbox_keypoints[2][1]>bbox_keypoints[0][1]:
            arctan1=(bbox_keypoints[2][1]-bbox_keypoints[0][1])/abs(bbox_keypoints[2][0]-bbox_keypoints[0][0])
        else:
            arctan1=abs(bbox_keypoints[0][1]-bbox_keypoints[2][1])/abs(bbox_keypoints[2][0]-bbox_keypoints[0][0])
  
        alpha=math.atan(arctan1)

        if bbox_keypoints[2][1]>bbox_keypoints[0][1]:
            alpha=-alpha
        else:
            alpha=alpha
        alpha_string = 'alpha  {:.2f}'.format(alpha) 
        img_bgr = cv2.putText(img_bgr, alpha_string, (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)
        
        
        #line 2
        ptStart = (bbox_keypoints[1][0], bbox_keypoints[1][1])
        ptEnd = (bbox_keypoints[4][0], bbox_keypoints[4][1])
        point_color = (0, 0, 255) # BGR
        thickness = 2 
        lineType = 4
        cv2.line(img_bgr, ptStart, ptEnd, point_color, thickness, lineType)
        arctan2=abs(bbox_keypoints[4][0]-bbox_keypoints[1][0])/abs(bbox_keypoints[4][1]-bbox_keypoints[1][1])
        beta=math.atan(arctan2)
        beta_string = 'beta  {:.2f}'.format(beta-0.50) # minus calibrated value
        img_bgr = cv2.putText(img_bgr, beta_string, (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)

        
        if alpha >0.3:
            warning_string = 'Are you tired?' # 写在画面上的字符串
            img_bgr = cv2.putText(img_bgr, warning_string, (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
        if beta-0.6 <-0.2:
            warning_string = 'Are you tired?' # 写在画面上的字符串
            img_bgr = cv2.putText(img_bgr, warning_string, (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    
    # record the time when processing finished
    end_time = time.time()
    # calculate FPS
    FPS = 1/(end_time - start_time)

    FPS_string = 'FPS  {:.2f}'.format(FPS) 
    img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)

    
    return img_bgr, alpha_string, beta_string


all_records = list()

# get camera index
cap = cv2.VideoCapture(1)

# open cap
cap.open(1)

# loop until brake
while cap.isOpened():
    
    # get frame
    success, frame = cap.read()
    
    if not success: # quit if no camera
        print('cannot find the camera')
        break
    
    # process each frame
    frame, alpha, beta = process_frame(frame)
    
    now = datetime.datetime.now()
    one_record = now.strftime("%Y-%m-%d %H:%M:%S") + "    " + alpha + "    " + beta
    all_records.append(one_record)
    
    # processed frame
    cv2.imshow('Driver monitoring system V1.0',frame)
    
    key_pressed = cv2.waitKey(60) # each () ms, which key has been tabed
    # print('keyborad：', key_pressed)

    if key_pressed in [ord('q'),27]: # q or esc to stop
        break
    
# close camera
cap.release()

# close window
cv2.destroyAllWindows()

# save data in txt
with open("record.txt", "w") as f:
    for one in all_records:
        f.write(one + "\n")
    
    f.close()

