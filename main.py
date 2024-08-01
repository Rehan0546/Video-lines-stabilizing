# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:41:32 2024

@author: rehan
"""
 
import pandas as pd

parquet_file_path = 'video_1_inferences.parquet'
df = pd.read_parquet(parquet_file_path)

def inference_from_file(frame_index,df=df):
    frame_index += 1
    return df.loc[frame_index]['bbox_xyxy']

import cv2, os
import numpy as np
import time

# feature detector module
orb = cv2.SIFT_create()

 # Shrink frame to lower size to reduce processing.
 # lower values has less time but shaking of the lines increase #higher values has more time but less shaking in lines.
scale_factor = 1

coordinates = {
                1: [[208, 276, 208, 368], [806, 208, 700, 208], [818, 268, 908, 268], [1514, 242, 1428, 194], [1526, 200, 1622,
                    166], [1734, 264, 1740, 382], [1630, 398, 1634, 544], [366, 396, 364, 534], [400, 582, 396, 728], [1034, 576,
                    1038, 712], [1596, 578, 1602, 728], [1488, 876, 1606, 812], [764, 816, 858, 816], [650, 734, 758, 738]]
                ,2: [[540, 314, 462, 388], [584, 454, 824, 432], [938, 414, 1056, 442], [1106, 450, 1266, 490], [564, 646, 544, 838],
                    [544, 838, 730, 870], [730, 870, 832, 710], [832, 716, 980, 838], [980, 838, 1024, 604], [1220, 696, 1462, 780],
                    [1530, 818, 1334, 1066]]
                ,3: [[150, 368, 480, 370], [510, 426, 428, 618], [416, 638, 834, 624], [1152, 448, 1226, 592], [1340, 442, 1588,
                    440], [1668, 272, 1784, 246], [390, 904, 62, 972]]
                ,4: [[812, 862, 874, 862], [876, 826, 1004, 828], [1042, 876, 1122, 880], [1146, 834, 1238, 836], [812, 248, 878,
                    248], [896, 196, 1008, 198], [1048, 232, 1130, 234], [1180, 208, 1236, 208], [1628, 132, 1608, 156], [1672,
                    194, 1644, 218], [1478, 264, 1454, 294], [1752, 410, 1736, 448], [1398, 364, 1440, 448], [1552, 642, 1532,
                    726], [1782, 596, 1798, 634], [1454, 792, 1490, 826], [1686, 522, 1754, 522], [474, 866, 434, 898], [472, 664,
                    514, 730], [532, 316, 494, 354], [418, 224, 458, 254], [312, 192, 322, 234], [198, 248, 178, 286], [144, 498,
                    104, 514], [174, 410, 240, 434]]
                ,5: [[558, 312, 556, 402], [574, 468, 576, 570], [540, 622, 542, 752], [1050, 628, 1054, 750], [1544, 632, 1554,
                    758], [1564, 470, 1568, 574], [1606, 436, 1610, 474], [1736, 324, 1738, 420], [1612, 244, 1560, 266], [1468,
                    288, 1420, 254], [810, 256, 902, 256], [688, 222, 796, 218], [648, 796, 754, 798], [764, 838, 856, 838]]
                  }


# to look only lines area, margin variable used to copy the orignal image around line                                                                                                                           
def create_blocks(frame, line_coordinates, margin = int(50)):
    
    black_frame = np.zeros_like(frame)

    for line in line_coordinates:
        x1, y1, x2, y2 = map(int, line)
        
        x1,x2 = min(x1,x2), max(x1,x2)
        y1,y2 = min(y1,y2), max(y1,y2)

        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin

        black_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2].copy()
    # cv2.imshow('ssss',black_frame)
    return black_frame


 # Shrink frame to lower size to reduce processing.
def shrink(frame, lines, scale_factor=scale_factor):

    
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    # Resize the coordinates of the lines
    lines = np.array([(int(x1 * scale_factor), int(y1 * scale_factor), 
                      int(x2 * scale_factor), int(y2 * scale_factor))
                     for (x1, y1, x2, y2) in lines] , dtype=np.float32)

    return frame, lines

# Resize the coordinates of the lines back to the original size
def unshrink(lines, scale_factor):
    
    lines = np.array([(int(x1 /scale_factor),
                       int(y1 / scale_factor),
                       int(x2 / scale_factor),
                       int(y2 / scale_factor))
                      for (x1, y1, x2, y2) in lines])

    return  lines

# function to get new lines

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Use FLANN to find matches
flann = cv2.FlannBasedMatcher(index_params, search_params)

def line_stabilize(frame,prev_line,kp1,des1,
                   reduced_matches = 1,
                   bf = flann):

    kp2, des2 = feature_detector(frame,prev_line,orb)
    if not kp2:
        return prev_line, kp1, des1
    # Perform feature matching
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        # if m.distance < reduced_matches * n.distance:
            good_matches.append(m)

    # Get matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    # print('src_pts',src_pts.shape)
    # print('dst_pts',dst_pts.shape)
    if dst_pts.shape[0]>4:
        # Calculate homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3)
    
        # Apply homography to update line 

        new_lines = cv2.perspectiveTransform(prev_line.reshape(-1, 1, 2), H).reshape(-1,4)
        return new_lines, kp2, des2
    return prev_line, kp1, des1

# Feature extractor around lines
def feature_detector(frame,prev_line,orb):
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = create_blocks(frame, prev_line)
    # Find keypoints and descriptors in the frame
    kp1, des1 = orb.detectAndCompute(frame, None)
    return kp1, des1


# Run on the video
def video_line_stabelize(video_path,
                         show = True,
                         save = True,
                         coordinates = coordinates,
                         calibration_frame_period = 3):
    
    frame_num = 0
    # Read the video
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    
    
    
    if save:
        # Get video properties (width, height, and frames per second)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Define the codec and create a VideoWriter object for the output video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose other codecs like 'MJPG' or 'H264'
        output_video_path = f'output_{os.path.basename(video_path).split(".")[0]}.avi'
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
    
    
    # here it's getting the lines coordinates from dict
    prev_line = coordinates[int([i for i in video_name if i.isnumeric()][0])]
    
    # Read the first frame
    ret, frame = cap.read()
    
    ###################################################
    """
    Here is your detector function in which detected xyxy
    """

    xyxy = inference_from_file(frame_num)
    for j in xyxy:
        x1,y1,x2,y2 = j
        box_margin = 20
        x1 -= box_margin
        y1 -= box_margin
        x2 += box_margin
        y2 += box_margin
        # cv2.rectangle(frame,(x1,y1),(x2,y2), (0, 255, 0), 2)
        frame[y1:y2, x1:x2] = np.zeros_like(frame[y1:y2, x1:x2]) 
    """
    Here is your detector function in which detected xyxy
    """
    ############################################
    
    
    
    # original_shape = frame.shape[:2]
    frame, prev_line = shrink(frame, prev_line)
    kp1, des1 = feature_detector(frame,prev_line,orb)
    
    
    while True:
        # Read the current frame
        ret, frame = cap.read()
        
        start_time = time.time()
        if not ret:
            break            
        frame_ori = frame.copy()
        
        
        ###################################################
        """
        Here is your detector function in which detected xyxy
        """
        try:
            xyxy = inference_from_file(frame_num)
            for j in xyxy:
                x1,y1,x2,y2 = j
                box_margin = 20
                x1 -= box_margin
                y1 -= box_margin
                x2 += box_margin
                y2 += box_margin
                # cv2.rectangle(frame,(x1,y1),(x2,y2), (0, 255, 0), 2)
                frame[y1:y2, x1:x2] = np.zeros_like(frame[y1:y2, x1:x2]) 
        except:
            pass
        """
        Here is your detector function in which detected xyxy
        """
        ############################################
        
        
    
        # This if statement will decide when to recalibrate the lines with previous frames, For the false negative adjustments
        if frame_num==0:
            prev_line,kp1, des1 = line_stabilize(cv2.resize(frame, None, fx=scale_factor, fy=scale_factor),
                                                 prev_line,kp1,des1)
            prev_line_,kp1_, des1_ = prev_line.copy(), kp1[:], des1[:]
        
        elif frame_num%calibration_frame_period==0 and frame_num>0:
            prev_line_,kp1_, des1_ = line_stabilize(cv2.resize(frame, None, fx=scale_factor, fy=scale_factor),
                                                  prev_line_,kp1_, des1_)
            prev_line,kp1, des1 = prev_line_.copy(), kp1_[:], des1_[:]
        else:
            prev_line,kp1, des1 = line_stabilize(cv2.resize(frame, None, fx=scale_factor, fy=scale_factor),
                                                 prev_line,kp1,des1)
            
        frame_num +=1
        lines = unshrink(prev_line, scale_factor)
        # Draw the stabilized lines on the current frame
        # print(lines)
        for line in lines:
            x1, y1, x2, y2 = line.ravel()
            cv2.line(frame_ori, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
        cv2.putText(frame_ori, f'frame:{frame_num} {time.time()-start_time:.4f}s', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        # Display the result
        if show:
            cv2.imshow('Stabilized Lines', cv2.resize(frame_ori,(720,480)))
            
            if cv2.waitKey(30) & 0xFF == 27:
                break
        if save:
            out.write(frame_ori)
        
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    if save:
        out.release()

video_line_stabelize('video_1.mp4',
                         show = True)