# Vehicle Classification and Detection

In this project, I have implemented 2 methodologies in classifying and detecting the vehicles in an given video frame. 
Method 1 uses just the image processing techniques
1. Gaussian Mixture-based Background/Foreground Segmentation Algorithm 
2. Morphological transformations (erosion followed by dilation)

https://docs.opencv.org/4.x/opening.png![image](https://user-images.githubusercontent.com/91593176/214158285-3102604b-1593-4202-8f5f-051fe5826cad.png)

Method 2 uses COCO dataset (vehicle classes) + pretrained YOLOv3 weights, Non-Max Suppression is used to avoid many detections while using yolo. 

Conclusions: 
In short, the image processing techniques proved to be much faster rather than using the deep learning based techniques, but the scalling to a 
multiple classification model is easier and better in deep learning based methods. 

References: 
https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
