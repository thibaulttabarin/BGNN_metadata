import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import imutils
from statistics import mode

color_img = cv2.imread('mask.jpg')
orig_img = cv2.imread('/home/HDD/bgnn_data/other_museums/osum/ruler2.png',0)
#img = orig_img
#print(img)
img = cv2.Canny(orig_img,100,200)
cv2.imwrite("canny_ruler.png", img)
exit(0)

y, x = np.nonzero(img)

x = x - np.mean(x)
y = y - np.mean(y)
coords = np.vstack([x, y])

cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)

sort_indices = np.argsort(evals)[::-1]
x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
x_v2, y_v2 = evecs[:, sort_indices[1]]

scale = 50.0
plt.plot(x, y, 'k.')
plt.plot([x_v1*-scale*2, x_v1*scale*2],
         [y_v1*-scale*2, y_v1*scale*2], color='red')
plt.plot([x_v2*-scale, x_v2*scale],
         [y_v2*-scale, y_v2*scale], color='blue')
plt.axis('equal')
plt.gca().invert_yaxis()  # Match the image system with origin at top left

"""
base = np.array([1,0])
eigen = np.array([x_v1,y_v1])
angle = math.acos(np.dot(eigen,base)) * 180.0 / math.pi

# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
print(angle)
#exit(0)
rotated = imutils.rotate_bound(color_img, 0)
#cv2.imshow("Rotated (Correct)", rotated)
#cv2.waitKey(0)

edges_2 = cv2.Canny(rotated,100,200)
plt.subplot(),plt.imshow(edges_2,cmap = 'gray')

#counts = []

#for i in range(300):
#    counts.append(np.count_nonzero(edges_2[i]))
#    print(f'{i}: {counts[-1]}')

#print(f'Mode: {mode(counts)}')
"""

plt.show()
