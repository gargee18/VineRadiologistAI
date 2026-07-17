import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

#Load Image
volume = tiff.imread("/home/phukon/code_python/2DProjection/test/RAW_sr/CEP020_APO1_RX.tif").astype(np.float32)

volume = volume / 1000.0

print("Volume shape:", volume.shape)

# Beer-Lambert Projection
I0 = 1.0
attenuation_scale = 0.015
# Sum along Z (axis=0)
sum_mu = np.sum(volume*attenuation_scale, axis=0)

projection = I0 * np.exp(-sum_mu)

# Middle slice for visualization
z_mid = volume.shape[0] // 2
middle_slice = volume[z_mid, :, :]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(middle_slice, cmap='gray')
axes[0].set_title("Middle Slice (Z = {})".format(z_mid))
axes[0].axis("off")

axes[1].imshow(projection, cmap='gray')
axes[1].set_title("Simulated Radiograph")
axes[1].axis("off")

print("Projection shape:", projection.shape)
plt.tight_layout()
plt.show()
#tiff.imwrite("/home/phukon/code_python/2DProjection/test/PROJ/CEP020_APO1_RX_PROJ_SIM.tif",projection)
#print("Projection saved successfully.")