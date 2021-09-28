# USAGE
# python3 detect_faces.py --dir folder

# adapted from https://github.com/jbrownlee/mtcnn

import numpy as np
import argparse
import os
import cv2
from imutils import paths
import time
import mtcnn



# output folder for the CSV and img files
output = "OUT_csv"
outputImg = "OUT_img"
nbFaces = 0

# detecting  homothetic contours
homotheticThreshold = 1.3 # 30%  tolerance
areaThreshold =1.4 # 40%  tolerance


def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0:
        print ("\tno")
        return False
    print ("\tyes")
    return (x, y, w, h)

# test if the bounding box is homothetic to the source image and has a similar size
def homothetic(c1,c2,area1,area2):  # (x,y,w,h)
    ratio1 = float(c1[2]) / float(c1[3])
    ratio2 = float(c2[2]) / float(c2[3])
    tmp=max(ratio1,ratio2)/min(ratio1,ratio2)
    print ("ratio area:%f" % (area1/area2))
    print ("ratio w: %f - ratio h : %f - max-min : %f" % (ratio1, ratio2, tmp))
    if tmp < homotheticThreshold and ((area1/area2) < areaThreshold):
        #print ("\thomothetic!")
        return True
    else:
        #print ("\tnon-homothetique")
        return False

# faces detection
def process_image(file):
	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it


	image = cv2.imread(file)
	(h, w) = image.shape[:2]
	areaImg = h*w
	#blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
	outText=""

	global nbFaces
	detections = detector.detect_faces(image)

	# loop over the detections
	for i in range(0, len(detections)):
			# extract the confidence (i.e., probability) associated with the prediction
			confidence = detections[i]['confidence']
			# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
			if (confidence > args["confidence"]):
				# compute the (x, y)-coordinates of the bounding box for the object
				box = detections[i]['box']
				print (box)
				#(startX, startY, endX, endY) = box
				wBox = box[2]
				hBox = box[3]
				nbFaces += 1
				text = "{:.2f}%".format(confidence * 100)
				#print (f"\t {text}")
				#print (startX, startY,(endX-startX),(endY-startY))
				# draw the boxes
				cv2.rectangle(image, (box[0], box[1]),(box[0]+wBox,box[1]+hBox), (0,0,255),2)
				#cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
				cv2.putText(image, text, (box[0], box[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
				# build the data
				if (outText ==""):
					outText = "face,%d,%d,%d,%d,%.2f" % (box[0], box[1],wBox,hBox, confidence)
				else:
					outText = "%s_face,%d,%d,%d,%d,%.2f" % (outText, box[0],box[1],wBox,hBox,confidence)

	if outText != "":
		print (outText)
		# open output file
		filename = os.path.splitext(os.path.basename(file))[0]
		outPath = os.path.join(outputDir, f"{filename}.csv" )
		outFile = open(outPath,"w")
		print ("%s;%s" % (filename, outText), file=outFile)
		outFile.close()
		# show the output image
		#cv2.imshow("Output", image)
		#cv2.waitKey(0)
		# save output image with annotations
		outPath = os.path.join(outputImgDir, f"{filename}.jpg" )
		cv2.imwrite(outPath, image)
	else:
		print ("\tno detection")


###################################
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dir", required=True,
	help="path to input image folder")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
	help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

outputDir = os.path.realpath(output)
if not os.path.isdir(outputDir):
	ap.error(f"Output .csv directory {outputDir} does not exist")
else:
	print (f"CSV files will be saved to {outputDir}")

outputImgDir = os.path.realpath(outputImg)
if not os.path.isdir(outputImgDir):
	ap.error(f"\nOutput image directory {outputImgDir} does not exist")
else:
	print (f"Output images will be saved to {outputImgDir}")

# load our serialized model from disk
print(" loading model...\n")
from mtcnn.mtcnn import MTCNN
import cv2
detector = MTCNN()

# load the images list
filePaths = list(paths.list_images(args["dir"]))
filePaths = [img.replace("\\", "") for img in filePaths]

start = time.time()
for i in filePaths:
	print (f"...analysing image {i}")
	filename = os.path.splitext(os.path.basename(i))[0]
	outPath = os.path.join(outputDir, "%s.csv" % filename)
	# don't reprocess an existing result
	if  os.path.isfile(outPath):
		print(" data file for %s already exists" % filename)
	else:
		try:
			process_image(i)
		except AttributeError as exc:
			print("Unexpected error:", exc)
end = time.time()

fps_label = "\n   -> time: %.2f (%.2f image/s)" % ((end - start),(end - start)/len(filePaths))
print (fps_label)

print (f"\n ### faces: {nbFaces} ###")
print (f" ### images analysed: {len(filePaths)} ###")
