from sklearn.externals import joblib
from PIL import Image
import numpy as np

import cv2 as cv
import matplotlib.pyplot as plt

model = joblib.load("svc_cls.pkl")

img = Image.open('../images/4.png').convert("L")
img = img.resize((28,28))
img = np.array(img)
redata = img.reshape(1, -1)

#img = plt.imread('../images/4.png')
#img = cv.resize(img, (28, 28))
#cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
#redata = np.array(img).reshape(1, -1)
print(img)
print("\n")
print(redata)

predicted = model.predict(redata)
print(predicted)