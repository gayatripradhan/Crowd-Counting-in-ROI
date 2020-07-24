# USAGE
# python peopleCount.py --input peopleWalking.mp4 --output output/peopleCount_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-l", "--lable", required=True,
	help="object to detect")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

object_name = args["lable"]
# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

frame_number= 0

bol,first = vs.read()
showCrosshair = False
roi = cv2.selectROI(first,showCrosshair)
cv2.waitKey(0)
cv2.destroyAllWindows()

roi_x,roi_y,roi_w,roi_h = roi

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    cv2.rectangle(img=frame, pt1=(roi_x, roi_y), pt2=(roi_x + roi_w, roi_y + roi_h), color=(0, 0, 255), thickness=2)
    
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    centers = []
    people_count=0
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                centers.append([centerX,centerY])
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])
    
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center_x,center_y = centers[i][0],centers[i][1]
            
            if LABELS[classIDs[i]]== object_name:
                if((roi_x < center_x < (roi_x + roi_w)) and (roi_y < center_y < (roi_y + roi_h))):
                    # draw a bounding box rectangle and label on the frame
                    people_count = people_count + 1
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    
                    cv2.circle(frame, (center_x, center_y), 5, (255,255,255), -1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                    
                    cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.putText(frame, "Total no. of " + str(object_name) + " in ROI: " + str(people_count), (40, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
    cv2.imshow("image",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
    
#    if writer is None:
#        # initialize our video writer
#        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#        writer = cv2.VideoWriter(args["output"], fourcc, 30,
#			(frame.shape[1], frame.shape[0]), True)
#        
#        # some information on processing single frame
#        if total > 0:
#            elap = (end - start)
#            print("[INFO] single frame took {:.4f} seconds".format(elap))
#            print("[INFO] estimated total time to finish: {:.4f}".format(
#				elap * total))
#    print(frame_number)
#    frame_number=frame_number+1
#
#	# write the output frame to disk
#    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()