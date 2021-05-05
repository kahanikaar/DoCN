import os
import sys
import cv2
import imutils
import numpy as np
from imutils import perspective
import matplotlib.pyplot as plt
from rembg.bg import remove as rembg
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


APPROX_POLY_DP_ACCURACY_RATIO = 0.02
IMG_RESIZE_H = 500.0



def scan(image):
    data=np.fromfile(image)
    bytes = np.frombuffer(rembg(data), np.uint8)

    img = cv2.imdecode(bytes, cv2.IMREAD_UNCHANGED)
    orig = img.copy()
    orig1=img.copy()

    ratio = img.shape[0] / IMG_RESIZE_H

    img = imutils.resize(img, height=int(IMG_RESIZE_H))
    orig = imutils.resize(orig, height=int(IMG_RESIZE_H))
    _, img = cv2.threshold(img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    img = cv2.medianBlur(img, 15)

    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    

    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, APPROX_POLY_DP_ACCURACY_RATIO * perimeter, True)

        if len(polygon) == 4:
            outline = polygon.reshape(4, 2)
            target=polygon
   
    cv2.drawContours(orig, [target], -1, (0, 255, 0), 2)
    
    plt.figure(5, figsize=(7,7))
    plt.imshow(orig, cmap='gray')
    plt.show()
    if outline is None:
        boudingBox = orig
        crop=orig1
    else:
        orig = imutils.resize(orig, height=int(IMG_RESIZE_H*ratio))  
        boundingBox = perspective.four_point_transform(orig, outline*ratio )
        crop=perspective.four_point_transform(orig1, outline*ratio )
    

    image=os.path.splitext(os.path.basename(image))[0]

    cv2.imwrite("./Outputs/"+image+"_BoundingBox.jpeg", boundingBox)
    cv2.imwrite("./Outputs/"+image+"_Crop.jpeg", crop)    
    
    
if __name__=='__main__':
    scan(sys.argv[1])
