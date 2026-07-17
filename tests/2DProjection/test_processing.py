import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

img = tiff.imread("/home/phukon/code_python/2DProjection/test/PROJ/CEP020_APO1_RX_PROJ_IJ.tif")

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
enhanced_img = clahe.apply(img)


ct_volume = tiff.imread("/home/phukon/code_python/2DProjection/test/RAW_sr/CEP020_APO1_RX.tif")
rotated_vol = rotate(ct_volume, angle=90, axes=(0,2), reshape=False)
drr = np.sum(rotated_vol, axis=0)

drr_norm = cv2.normalize(drr, None, 0, 255, cv2.NORM_MINMAX)
drr_uint8 = drr_norm.astype(np.uint8)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
drr_clahe = clahe.apply(drr_uint8)

fig, axs = plt.subplots(1, 3, figsize=(12, 5))

axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original")
axs[1].imshow(enhanced_img, cmap='gray')
axs[1].set_title("CLAHE Enhanced")
axs[2].imshow(drr_clahe, cmap='gray')
axs[2].set_title("Enhanced + 90° rotation")

plt.tight_layout()
plt.show()


