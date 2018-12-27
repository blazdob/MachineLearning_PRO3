from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import numpy as np

digits = datasets.load_digits()
features = digits.data
labels = digits.target

clf = SVC(gamma = 0.001)
clf.fit(features, labels)


img = misc.imread("number.jpg")
img = misc.imresize(img, (8,8))
img = img.astype(digits.images.dtype)
img = misc.bytescale(img, high = 16, low = 0)

x_test = list()

for vrstica in img:
    for piksel in vrstica:
        x_test.append(sum(piksel)/3.0)


print(clf.predict([x_test]))