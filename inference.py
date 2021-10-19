import cv2
from PIL import Image
from svm_predict import *
import pandas

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]
model.conf = 0.3
# Images
for f in ['zidane.jpg', 'bus.jpg']:
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
img1 = Image.open('zidane.jpg')  # PIL image
img1 = cv2.imread('zidane.jpg')
img2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]  # batch of images

# Inference
results = model(img1)  # includes NMS

# Results
results = xyxy2xywh(results.xyxy[0])
r_list = results.tolist() # turn prediction tensor to list
#print(r_list[0][0], type(r_list[0][0]))
event, proba, predictions = predict(r_list)
print(event,proba,predictions)
imgcv = drawBoundingBox(img1, r_list, predictions, proba)

cv2.imshow("frame", imgcv)
cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()
#print(results.pandas().xyxy[0]) # img1 predictions (pandas)