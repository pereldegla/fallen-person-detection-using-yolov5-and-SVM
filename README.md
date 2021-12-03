# Fallen person detection using yolov3 and SVM
SVM-classification approach based on : https://www.mdpi.com/2079-9292/8/9/915/html
![image_0008](https://user-images.githubusercontent.com/46407601/144666263-20642623-4a28-41b3-9a12-df1f718f8ee3.jpg)

---
### Table of contents 

- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)
- [License](#License)
- [Author Info](#autor-info)

---

## Description 

The goal here is to develop a reliable fall detection system. As it is one of the most dangerous issues for the elderly population. It is source of injury, loss of mobility, fear of falling and even death... 
We are not preventing the fall but increasing our reactivity to it. Using person detection and fall classification this approach solves the fall detection problem from end-to-end. The person detection algorithm aims to localize all persons in an image. Its output is the enclosing bounding boxes and the confidence scores that reflect how likely it is that the boxes contain a person. Fall classification estimates if the detected person is in a fall or not. The model is a SVM training on the Fallen Person Dataset (FPDS).

FPDS dataset is public and available at http://agamenon.tsc.uah.es/Investigacion/gram/papers/fall_detection/FPDS_dataset.zip.
The training process is coming soon on the training branch of the repository.

#### Technologies
- Yolov3
- Sklearn

---

## How to use
