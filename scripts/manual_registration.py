import imagej
from scyjava import jimport
import numpy as np

FIJI_PATH = "/home/phukon/Fiji.app"

CT_PATH = "/home/phukon/Desktop/Ct_PXR_manual_registration/CEP_1191_2026_XR.tif"

PXR_PATH = "/home/phukon/Desktop/Ct_PXR_manual_registration/CEP_1191_2026_PXR.tif"

TRANSFORM_PATH = "/home/phukon/Desktop/Ct_PXR_manual_registration/CEP_1191_transform_test.txt"

# Start Fiji
ij = imagej.init(FIJI_PATH, mode="interactive")

# Fiji classes
IJ = jimport("ij.IJ")
Image3DUniverse = jimport("ij3d.Image3DUniverse")
StackConverter = jimport("ij.process.StackConverter")
Transform3D = jimport("org.jogamp.java3d.Transform3D")
Matrix4d = jimport("org.jogamp.vecmath.Matrix4d")


# -------------------------
# Load CT
# -------------------------
ct = IJ.openImage(CT_PATH)

cal = ct.getCalibration()

print("FIJI pixel width :", cal.pixelWidth)
print("FIJI pixel height:", cal.pixelHeight)
print("FIJI pixel depth :", cal.pixelDepth)
print("FIJI unit        :", cal.getUnit())

print("Original CT bit depth:", ct.getBitDepth())


# Brightness / contrast window for viewing
ct.setDisplayRange(20, 800)

# Convert CT stack to 8-bit
StackConverter(ct).convertToGray8()

print("Viewer CT bit depth:", ct.getBitDepth())


# -------------------------
# Load PXR
# -------------------------
pxr = IJ.openImage(PXR_PATH)
pxr.show()


# -------------------------
# Open 3D Viewer
# -------------------------
univ = Image3DUniverse()
univ.show()

content = univ.addVoltex(ct)
content.setThreshold(1)
content.setTransparency(0.0)

mid = ct.getNSlices() // 2
ct.setSlice(mid)

stats = ct.getStatistics()

print("min:", stats.min)
print("max:", stats.max)
print("mean:", stats.mean)

# Select CT object
univ.select(content)

print("Ready for manual rotation")

input("Rotate the CT to match the PXR, then press ENTER to SAVE the transform...")

# Get Fiji's translation
t_translate = Transform3D()
content.getLocalTranslate(t_translate)

# Get Fiji's rotation
t_rotate = Transform3D()
content.getLocalRotate(t_rotate)

# Fiji combines them as: translation * rotation
t_translate.mul(t_rotate)

# Convert Java Transform3D to a 4x4 NumPy matrix
m = Matrix4d()
t_translate.get(m)

T = np.array([
    [m.m00, m.m01, m.m02, m.m03],
    [m.m10, m.m11, m.m12, m.m13],
    [m.m20, m.m21, m.m22, m.m23],
    [m.m30, m.m31, m.m32, m.m33],
], dtype=float)

# Save
np.savetxt(TRANSFORM_PATH, T, fmt="%.12f")

print("\nSaved transform:")
print(T)
print("\nSaved to:", TRANSFORM_PATH)

# Close windows
try:
    univ.close()
except:
    pass

try:
    pxr.close()
except:
    pass

try:
    ct.close()
except:
    pass
