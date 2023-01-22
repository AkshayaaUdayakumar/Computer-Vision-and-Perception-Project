import cv2
import numpy as np

class Car:
    tracks=[]
    def __init__(self,i,xi,yi,max_age):
        self.i=i
        self.x=xi
        self.y=yi
        self.tracks=[]
        self.done=False
        self.age=0
        self.max_age=max_age
        self.dir=None

    def getId(self):
        return self.i

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def timedOut(self):
        return self.done

    def age_one(self):
        self.age+=1
        if self.age>self.max_age:
            self.done=True
        return  True

cars = []
max_p_age = 10
pid = 1
Up_line=400
down_line=250
up_limit=230
down_limit=int(4.5*(500/5))

backsub=cv2.createBackgroundSubtractorMOG2(detectShadows=False,history=200,varThreshold = 90)
cap=cv2.VideoCapture("video.mp4") 

while(True): 
    ret, frame=cap.read() 
    frame=cv2.resize(frame,(1000,600))
    for i in cars:
        i.age_one()
    bgmask=backsub.apply(frame)

    if ret==True:
        ret,imBin=cv2.threshold(bgmask,200,255,cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, np.ones((3,3),np.uint8)) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((6,6),np.uint8))
        (contours0,hierarchy)=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
        for cnt in contours0:
            area=cv2.contourArea(cnt)
            if area>300:
                m=cv2.moments(cnt) #get weighted average of image pixel intensities
                cx=int(m['m10']/m['m00'])
                cy=int(m['m01']/m['m00'])
                x,y,w,h=cv2.boundingRect(cnt) 
                new=True
                if cy in range(up_limit,down_limit):
                    for i in cars:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                            new = False
                            i.updateCoords(cx, cy)                          
                            break
                        if i.timedOut():
                            index=cars.index(i)
                            cars.pop(index)
                            del i
                    if new==True:
                        p=Car(pid,cx,cy,max_p_age)
                        cars.append(p)
                        pid+1
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        for i in cars:
            cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1, cv2.LINE_AA)
            if down_line+20<= i.getY() <= Up_line-20:
               a = (h + (0.75*w)- 100)
               if a >= 0:
                     cv2.putText(frame, "Truck", (i.getX(), i.getY()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
               else:
                     cv2.putText(frame, "Car", (i.getX(), i.getY()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        #Region of Interest display
        frame=cv2.line(frame,(0,Up_line),(1000,Up_line),(255,0,0),3,8) #for blue
        frame = cv2.line(frame, (0, down_line), (1000, down_line), (255, 0,0), 3, 8) 
        cv2.imshow('Capture',frame)
        if cv2.waitKey(1)&0xff==ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

