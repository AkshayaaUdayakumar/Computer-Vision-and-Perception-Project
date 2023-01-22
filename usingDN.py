'''
The script reads in a video file "video.mp4" and 
runs object detection on each frame. The function "Processing" is applied to the output
of the object detection, which filters the detections to only keep those that are of the 
required classes (indexed 2, 5, 7 in the "coco.names" file), 
and have a confidence score greater than 0.5. The function then applies non-maxima suppression
to the remaining detections and draws rectangles and labels around the objects in the image.
Finally, it returns the detection with some additional information. 
The objects detected are displayed using the cv2.imshow function, 
and the script terminates when the user presses "q".
'''

import cv2
import numpy as np

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def Processing(outputs, img):
    required_class_index = [2, 5, 7] #car,bus, truck
    height, width = img.shape[:2]
    boxes, classIds, confidence_scores, detection = [], [], [], []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index and confidence > 0.5:
                w, h = int(det[2]*width), int(det[3]*height)
                x, y = int((det[0]*width)-w/2), int((det[1]*height)-h/2)
                boxes.append([x, y, w, h])
                classIds.append(classId)
                confidence_scores.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            name = classes[classIds[i]]
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            detection.append([x, y, w, h, confidence, required_class_index.index(classIds[i])])
            print("Detections:", detection)
    return detection

net = cv2.dnn.readNetFromDarknet("yolov3-320.cfg", "yolov3-320.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
video = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = video.read()
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320,320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layer_names = net.getLayerNames()
    output_names = [(output_layer_names[i - 1]) for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_names)
    height, width, channels = frame.shape
    vehicle_detections = []
    vehicle_detections = Processing(detections, frame)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

