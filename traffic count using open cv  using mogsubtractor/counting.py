import cv2
import numpy as np

bgsMOG = cv2.createBackgroundSubtractorMOG2()
cap    = cv2.VideoCapture("video.avi")
counter = 0

if cap:
    while True:
        ret, frame = cap.read()
    
        if ret:            
            fgmask = bgsMOG.apply(frame, None, 0.01)
            cv2.line(frame,(100,50),(400,50),(200,255,6),2)
            # To find the countours of the Cars
            _, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            try:
                hierarchy = hierarchy[0]
            except:
                hierarchy = []

            for contour, hier in zip(contours, hierarchy):
                (x, y, w, h) = cv2.boundingRect(contour)

                if w > 20 and h > 20:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 1)

                    #To find centroid of the Car
                    x1 = w/2      
                    y1 = h/2

                    cx = x+x1
                    cy = y+y1
                    centroid = (cx,cy)
                    
                    cv2.circle(frame,(int(cx),int(cy)),2,(0,0,255),-1)

                    if centroid > (23, 104) and centroid < (28.5, 140):
                        counter +=1
                    print(counter)
                    shape = frame.shape
                    size = frame.size
                    
                    cv2.putText(frame, str(counter), (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.imshow('Output', frame)


            
            key = cv2.waitKey(60)
            if key == 27:
                break
print(counter)
cap.release()
cv2.destroyAllWindows()
