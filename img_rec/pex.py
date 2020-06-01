import sys
import cv2
import numpy as np
import imutils

def process(imgFile):
	gray = cv2.imread(imgFile, 0)
	_, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
	thresh = cv2.Canny(thresh, 30, 100)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	output = gray.copy()
	for i, c in enumerate(cnts):
		A = cv2.contourArea(c)
		if A < 150: continue
		print(A)
		cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
		cv2.imshow("Contours", output)
		cv2.waitKey(0)
		print('--'*50)
if __name__ == '__main__':
	#x = np.random.randint(100, size=(5,4,3))
	#print(x)
	#exit()
	assert(len(sys.argv) == 2)
	imgFile = sys.argv[1]
	process(imgFile)
