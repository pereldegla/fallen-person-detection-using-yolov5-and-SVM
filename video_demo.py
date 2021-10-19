from darkflow.net.build import TFNet
import cv2
from matplotlib.image import imread
import svm_predict
import os
import math 
import nms
import numpy as np
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
        samples=[[0,0,0,0]] #we start the samples container with zeros to avoid size    
        for box in result:
            # print(box)
            x1,y1,x2,y2 = (box['topleft']['x'],box['topleft']['y'],box['bottomright']['x'],box['bottomright']['y'])
            conf = box['confidence']
            # print(conf)
            label = box['label']
            if label != 'person':
                continue
            if conf < 0.4:
                continue
            samples=np.append(samples,[[x1,y1,x2,y2]],axis=0)                   
        return samples

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.4}

tfnet = TFNet(options)
cap=cv2.VideoCapture(0) #add filepath for video
# Read until video is completed
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame    
    result = tfnet.return_predict(frame)
    samples=get_samples(result)
    samples=np.delete(samples,0,0)
    samples=nms.non_max_suppression_fast(samples, 0.6)
    _,proba,predictions=svm_predict.predict(samples)
    frame=drawBoundingBox(frame,samples, predictions,proba)
    out.write(frame)
    #cv2.imshow("demo",frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(10) & 0xFF == ord('q'): #set waitKey to 25 for video
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
out.release()

