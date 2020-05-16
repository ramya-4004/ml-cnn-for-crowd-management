from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from centroidtracker import CentroidTracker

# initialize the HOG descriptor/person detector
# pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

ct = CentroidTracker()
cap = cv2.VideoCapture('campus4-c0.avi')

while True:
    ret, image = cap.read()
    image = imutils.resize(image, width=min(400, image.shape[1]))

    ####################--PEDESTRAIN DETECTOR--#####################

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # this helps in the combination of multiple boxes detected on a single pedestrian into a single boundary box
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    ####################################################################

    #########################--CENTROID TRACKER & ARCS OF CUSTOMERS--##############################
    objects = ct.update(pick)

    # centre of our subsequent arcs
    x = image.shape[1] // 2
    y = image.shape[0]
    major_axis = 40
    minor_axis = 20
    # loop over the tracked objects
    for (OID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(OID)
        # for i in range(len(OID)):
        #     cv2.ellipse(image, (x, y), (major_axis + (10*i) , minor_axis + (10*i)), angle=0, startAngle=180, endAngle=360, color=(0, 0, 255), thickness=3)
        cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    ##########################################################################

    ###############--DISPLAY--##########################
    cv2.ellipse(image, (x, y), (40, 20),  angle=0, startAngle=180, endAngle=360, color=(0, 0, 255), thickness=3)
    cv2.ellipse(image, (x, y), (60, 30), angle=0, startAngle=180, endAngle=360, color=(0, 0, 255), thickness=3)
    cv2.ellipse(image, (x, y), (80, 40), angle=0, startAngle=180, endAngle=360, color=(0, 0, 255), thickness=3)
    cv2.ellipse(image, (x, y), (100, 50), angle=0, startAngle=180, endAngle=360, color=(0, 0, 255), thickness=3)
    cv2.imshow("Live", image)
    if cv2.waitKey(1) & 0xFFF == 27:
        break
cv2.destroyAllWindows()
