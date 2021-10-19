import cv2
from matplotlib.image import imread
import svm_predict
import os
import math
import pathlib
import datetime
import platform
import numpy as np


def dist(a,b):
    return math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)
    
def drawBoundingBox(imgcv,samples,predictions,proba):
    color=[[0,0,255],[0,255,0]]
    for i in range(len(predictions)):
        if predictions[i]==1:
            c=color[0]
            predictions[i]='Fall'
        else:
            c=color[1]
            predictions[i]='No Fall'
        text=predictions[i]+"-"+"{:.2f}".format(proba[i]*100)+"%"
        cv2.rectangle(imgcv,(samples[i][0],samples[i][1]),(samples[i][2],samples[i][3]),c,6)
        labelSize=cv2.getTextSize(text,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
        _x1 = samples[i][0]
        _y1 = samples[i][1]#+int(labelSize[0][1]/2)
        _x2 = _x1+labelSize[0][0]
        _y2 = samples[i][1]-int(labelSize[0][1])
        cv2.rectangle(imgcv,(_x1,_y1),(_x2,_y2),c,cv2.FILLED)
        cv2.putText(imgcv,text,(samples[i][0],samples[i][1]),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    return imgcv 
    
def get_samples(result): 
    samples=[[0,0,0,0]] #we start the samples container with zeros to avoid size matters
    #edges of the image
    topleft=[0,0]
    topright=[IMG_W,0]
    bottomleft=[0,IMG_H]
    bottomright=[IMG_W,IMG_H]
    #doubt =[] #flag for relocation, to be sent as a ROS msg 
    for box in result:
        label = box['label']
        if label != 'person':
            continue
        x1,y1,x2,y2 = (box['topleft']['x'],box['topleft']['y'],box['bottomright']['x'],box['bottomright']['y'])
        conf = box['confidence']
        """if (dist(topleft,[x1,y1])<40) or (dist(topright,[x2,y1])<40) or (dist(bottomleft,[x1,y2])<40) or (dist(bottomright,[x2,y2])<40):
            doubt.append('doubt')
        else:
            doubt.append('no doubt')"""
        if conf < 0.3:
            continue         
        samples=np.append(samples,[[x1,y1,x2,y2]],axis=0)
        #samples.append([x1,y1,x2,y2])  
    #print(doubt)                 
    return samples
            	        
def optimize(img,result,tfnet,samples_v1):  
    #Need some improvements : handling containers shifting sizes as the number of detections can shift
    #Solution : setting container filled with zeros and looping through     
    img90=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img270=cv2.rotate(img90, cv2.ROTATE_180)    
    result90=tfnet.return_predict(img90) 
    result270=tfnet.return_predict(img270)
    samples90=get_samples(result90)
    samples270=get_samples(result270)            
    #reverting the rotation effect on the coordinates in the pictures
    for i in range(len(samples90)): 
        samples90[i]=[-samples90[i][1],samples90[i][0],-samples90[i][3],samples90[i][2]]
        iou90=intersection_over_union(samples_v1[i],samples90[i])
    for j in range(len(samples270)):
        samples270[i]=[samples270[i][1],-samples270[i][0],samples270[i][3],-samples270[i][2]]
        iou270=intersection_over_union(samples_v1,samples270)
    #checking correspondance between bounding boxes        
    if ((iou90 > iou270) and (iou90>0.1)): 
        return samples90
    if ((iou270 > iou90) and (iou270>0.1)): 
        return samples270    
                                         
options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.3}

tfnet = TFNet(options)
folder="/home/perel/FPDS/YOLO_Object_Detection/sample_img/test"
output_path="/home/perel/FPDS/YOLO_Object_Detection/sample_img/test_out"

for filename in os.listdir(folder):
    imgcv = imread(os.path.join(folder,filename))
    result = tfnet.return_predict(imgcv) 
    samples=get_samples(result)
    samples=np.delete(samples,0,0) #we delete the first row we've created 
    samples=nms.non_max_suppression_fast(samples, 0.5)
    proba,predictions=svm_predict.predict(samples)
    imgcv=drawBoundingBox(imgcv,samples, predictions,proba)
    cv2.imwrite(os.path.join(output_path ,filename), imgcv)
