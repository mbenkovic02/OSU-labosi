import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib import transforms

image = img.imread('road.jpg')
image = image.copy()

plt.figure()
plt.imshow(image, alpha=0.6)
plt.show()

width = len(image[0])
quarter_width = int(width/4)
plt.imshow(image[:, 1*quarter_width: 2*quarter_width , :])
plt.show()

rotated_image = np.rot90(image, 3)
plt.imshow(rotated_image)
plt.show()

flipped_image = np.flip(image, 0)
plt.imshow(flipped_image)
plt.show()
