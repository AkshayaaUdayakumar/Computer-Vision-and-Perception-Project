# Vehicle Classification and Detection

In this project, I have implemented 2 methodologies in classifying and detecting the vehicles in an given video frame. 
Basically, To address the problem of the small object detection and the multi-scale variation of the object, the road surface area was defined as a remote area and a proximal area. The two road areas of each frame were sequentially detected to obtain good vehicle detection results in the monitoring field. here are the 2 methods I have used: 

Method 1 uses just the image processing techniques
1. Gaussian Mixture-based Background/Foreground Segmentation Algorithm 
2. Morphological transformations (erosion followed by dilation)
<img width="473" alt="image" src="https://user-images.githubusercontent.com/91593176/214159551-1e824373-b46f-404d-8363-4327fed75702.png">

https://docs.opencv.org/4.x/opening.png![image](https://user-images.githubusercontent.com/91593176/214158285-3102604b-1593-4202-8f5f-051fe5826cad.png)

Method 2 uses COCO dataset (vehicle classes) + pretrained YOLOv3 weights, Non-Max Suppression is used to avoid many detections while using yolo. 

<img width="460" alt="image" src="https://user-images.githubusercontent.com/91593176/214159532-485ae6f2-38d8-47f2-97d7-9794f473c458.png">


Conclusions: 
In short, the image processing techniques proved to be much faster rather than using the deep learning based techniques, but the scalling to a 
multiple classification model is easier and better in deep learning based methods. 

References: 
https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
