import pylab as pl
import numpy as np
from matplotlib import pyplot as plt

from sklearn.externals import joblib
from skimage.feature import hog

try:
    import cv2
except ImportError:
    print("You must have OpenCV installed")

# Load the classifier
#clf = joblib.load("digits_cls.pkl")

model = joblib.load("linear_svc_cls.pkl")

def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
	"""Draws circular points on an image."""
	img = in_img.copy()

	# Dynamically change to a colour image if necessary
	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	for point in points:
		img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
	show_image(img)
	return img
	
def distance_between(p1, p2):
	"""Returns the scalar distance between two points"""
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))

#sort the corners to remap the image
def getOuterPoints(rcCorners):
	ar = [];
	ar.append(rcCorners[0,0,:]);
	ar.append(rcCorners[1,0,:]);
	ar.append(rcCorners[2,0,:]);
	ar.append(rcCorners[3,0,:]);
	
	x_sum = sum(rcCorners[x, 0, 0] for x in range(len(rcCorners)) ) / len(rcCorners)
	y_sum = sum(rcCorners[x, 0, 1] for x in range(len(rcCorners)) ) / len(rcCorners)
	
	def algo(v):
		return (math.atan2(v[0] - x_sum, v[1] - y_sum)
				+ 2 * math.pi) % 2*math.pi
		ar.sort(key=algo)
	return (  ar[3], ar[0], ar[1], ar[2])
	
def crop_and_warp_my(img, crop_rect):
	IMAGE_WIDHT = 12
	IMAGE_HEIGHT = 12
	SUDOKU_SIZE= 9
	N_MIN_ACTVE_PIXELS = 10

	#point to remap
	points1 = np.array([
						np.array([0.0,0.0] ,np.float32) + np.array([144,0], np.float32),
						np.array([0.0,0.0] ,np.float32),
						np.array([0.0,0.0] ,np.float32) + np.array([0.0,144], np.float32),
						np.array([0.0,0.0] ,np.float32) + np.array([144,144], np.float32),
						],np.float32)    
	#outerPoints = getOuterPoints(crop_rect)
	points2 = np.array([crop_rect[1],crop_rect[0],crop_rect[3],crop_rect[2]],np.float32)

	#Transformation matrix
	pers = cv2.getPerspectiveTransform(points2,  points1);

	#remap the image
	return cv2.warpPerspective(img, pers, (SUDOKU_SIZE*IMAGE_HEIGHT, SUDOKU_SIZE*IMAGE_WIDHT));

def crop_and_warp_my1(img, crop_rect):
	IMAGE_WIDHT = 16
	IMAGE_HEIGHT = 16
	SUDOKU_SIZE= 9
	N_MIN_ACTVE_PIXELS = 10
	
	"""Crops and warps a rectangular section from an image into a square of similar size."""

	# Rectangle described by top left, top right, bottom right and bottom left points
	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

	# Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	# Get the longest side in the rectangle
	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	# Describe a square with side of the calculated length, this is the new perspective we want to warp to
	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	# Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
	m = cv2.getPerspectiveTransform(src, dst)

	# Performs the transformation on the original image
	return cv2.warpPerspective(img, m, (SUDOKU_SIZE*IMAGE_HEIGHT, SUDOKU_SIZE*IMAGE_WIDHT))
	

def crop_and_warp(img, crop_rect):
	"""Crops and warps a rectangular section from an image into a square of similar size."""

	# Rectangle described by top left, top right, bottom right and bottom left points
	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

	# Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	# Get the longest side in the rectangle
	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	# Describe a square with side of the calculated length, this is the new perspective we want to warp to
	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	# Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
	m = cv2.getPerspectiveTransform(src, dst)

	# Performs the transformation on the original image
	return cv2.warpPerspective(img, m, (int(side), int(side)))
	
import operator

def find_corners_of_largest_polygon(img):
	"""Finds the 4 extreme corners of the largest contour in the image."""	
	contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
	contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
	polygon = contours[0]  # Largest image

	# Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
	# Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

	# Bottom-right point has the largest (x + y) value
	# Top-left has point smallest (x + y) value
	# Bottom-left point has smallest (x - y) value
	# Top-right point has largest (x - y) value
	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

	# Return an array of all 4 points using the indices
	# Each point is in its own array of one coordinate
	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

def pre_process_image(img, skip_dilate=False):
	"""Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

	# Gaussian blur with a kernal size (height, width) of 9.
	# Note that kernal sizes must be positive and odd and the kernel must be square.
	proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
	
	# Adaptive threshold using 11 nearest neighbour pixels
	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
	#show_img(proc)
	#proc = cv2.adaptiveThreshold(proc,255,1,1,11,15)
	# Invert colours, so gridlines have non-zero pixel values.
	# Necessary to dilate the image, otherwise will look like erosion instead.
	proc = cv2.bitwise_not(proc, proc)

	# Dilate the image to increase the size of the grid lines.
	if not skip_dilate:
		kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dtype=np.uint8)
		proc = cv2.dilate(proc, kernel)
	
	return proc
	
def show_digits(digits, colour=255):
	"""Shows list of 81 extracted digits in a grid format"""
	rows = []
	with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
	for i in range(9):
		row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
		rows.append(row)
	show_image(np.concatenate(rows))
	
def display_rects(in_img, rects, colour=255):
	"""Displays rectangles on the image."""
	img = in_img.copy()
	for rect in rects:
		img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
	show_image(img)
	return img

def show_image(img):
	"""Shows an image until any key is pressed"""
	cv2.imshow('image', img)  # Display the image
	cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
	cv2.destroyAllWindows()  # Close all windows
	
def infer_grid(img):
	"""Infers 81 cell grid from a square image."""
	squares = []
	side = img.shape[:1]
	side = side[0] / 9
	for i in range(9):
		for j in range(9):
			p1 = (i * side, j * side)  # Top left corner of a bounding box
			p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
			squares.append((p1, p2))
	return squares

def cut_from_rect(img, rect):
	"""Cuts a rectangle from an image using the top left and bottom right points."""
	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

def scale_and_centre(img, size, margin=0, background=0):
	"""Scales and centres an image onto a new background square."""
	h, w = img.shape[:2]
	#print(h)
	#print(w)
	#show_image(img)

	def centre_pad(length):
		"""Handles centering for a given length that may be odd or even."""
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	#img = cv2.resize(img, (w, h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
	return cv2.resize(img, (size, size))

def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
	"""
	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
	"""
	img = inp_img.copy()  # Copy the image, leaving the original untouched
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	# Loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# Only operate on light or white squares
			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
				area = cv2.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
					max_area = area[0]
					seed_point = (x, y)

	# Colour everything grey (compensates for features outside of our middle scanning range
	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv2.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

	# Highlight the main feature
	if all([p is not None for p in seed_point]):
		cv2.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
				cv2.floodFill(img, mask, (x, y), 0)

			# Find the bounding parameters
			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left-4, top-4], [right+4, bottom+4]]
	return img, np.array(bbox, dtype='float32'), seed_point

def extract_digit(img, rect, size):
	"""Extracts a digit (if one exists) from a Sudoku square."""
	digit = cut_from_rect(img, rect)  # Get the digit box from the whole square
	
	#print(digit)
	# Use fill feature finding to get the largest feature in middle of the box
	# Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
	h, w = digit.shape[:2]
	margin = int(np.mean([h, w])/ 2.5)
	#print(margin)
	_, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
	#print(bbox)
	digit = cut_from_rect(digit, bbox)
	print(digit)
	# Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
	w = bbox[1][0] - bbox[0][0]
	h = bbox[1][1] - bbox[0][1]
	#print(w)
	#print(h)
	#print(digit)
	#return digit
	# Ignore any small bounding boxes
	if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:	
		return scale_and_centre(digit, size, 4)
	else:
		return np.zeros((0, 0), np.uint8)	
		
def predict_digits(img, squares, size):
	"""Extracts digits from their cells and builds an array"""
	digits = []
	#show_img(img)
	img = pre_process_image(img.copy(), skip_dilate=True)
	#show_img(img)
	for square in squares:
		arr = extract_digit(img, square, size)
		#print(arr)
		#img = np.array(arr)
		#print(img)
		redata = arr.reshape(1, -1)			
		print(len(redata[0]))
		#redata = scale_and_centre(arr, size, 4)	
		#redata = arr.reshape(1, -1)
		#print(redata)
		#print(len(redata[0]))
		if len(redata[0]) != 0:				
			predicted = model.predict(redata)
			print(predicted)		
		#print(redata)
		#if len(arr) != 1: 
		#	arr = arr.resize((28,28))		
		#	arr = np.array(arr)
		#	arr = arr.reshape(1, -1)
		#	predicted = model.predict(arr)
		#	print(predicted)
	#return digits

def show_img(img):
	#show image
	thresh = cv2.adaptiveThreshold(img,255,1,1,11,15)
	_ = pl.imshow(thresh, cmap=pl.gray())
	_ = pl.axis("off")
	pl.show()

original = cv2.imread('../sample/2.png', cv2.IMREAD_GRAYSCALE)
#show_img(original)
processed = pre_process_image(original)
#show_img(processed)
corners = find_corners_of_largest_polygon(processed)
#display_points(processed, corners)
cropped = crop_and_warp(original, corners)

squares = infer_grid(cropped)
#display_rects(cropped, squares)

predict_digits(cropped, squares, 28)

#digits = get_digits(cropped, squares, 28)
#show_digits(digits)
