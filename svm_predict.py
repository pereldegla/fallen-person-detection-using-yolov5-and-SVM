import pickle
import torch
import cv2
IMG_W = 1280
IMG_H = 720

filename = 'finalized_model.sav'
clf = pickle.load(open(filename, 'rb'))

def predict(samples):
    predictions=[]  
    proba=[]
    event=""
    for i in range(len(samples)):
        w =abs(samples[i][2]-samples[i][0])
        h =abs(samples[i][3]-samples[i][1])
        AR =w/h  #Aspect ratio of bounding box
        NW =w/IMG_W #Normalized bounding box width
        NB =1-samples[i][3]/IMG_H #Normalized bounding box bottom coordinate
        proba_tmp = clf.predict_proba([[AR/10, NW, NB]])
        proba_tmp = max(proba_tmp[0])
        if proba_tmp > 0.7:
            proba.append(proba_tmp)
            tmp = clf.predict([[AR/10, NW, NB]])
            predictions.append(tmp)
            if tmp[0] == 1:
                event = "Fall"
            else:
                event = "Nothing to see here"
    return event,proba,predictions

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def drawBoundingBox(imgcv,samples,predictions,proba):
    color=[[0,0,255],[0,255,0]]
    for i in range(len(predictions)):
        if predictions[i]==1:
            c=color[0]
            predictions[i]='Fall'
        else:
            c=color[1]
            predictions[i]='No Fall'
        text = predictions[i]+"-"+"{:.2f}".format(proba[i]*100)+"%"
        cv2.rectangle(imgcv,(int(samples[i][0]),int(samples[i][1])),(int(samples[i][2]),int(samples[i][3])),c,6)
        labelSize = cv2.getTextSize(text,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
        _x1 = samples[i][0]
        _y1 = samples[i][1]#+int(labelSize[0][1]/2)
        _x2 = _x1+labelSize[0][0]
        _y2 = samples[i][1]-int(labelSize[0][1])
        cv2.rectangle(imgcv,(int(_x1),int(_y1)),(int(_x2),int(_y2)),c,cv2.FILLED)
        cv2.putText(imgcv,text,(int(samples[i][0]),int(samples[i][1])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    return imgcv