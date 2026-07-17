
import os
import tifffile as tiff

base_root = "/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/2DProjection/Dataset_Vitimage2019/"

for specimen in os.listdir(base_root):
    seg_path = os.path.join(base_root, specimen, "SEG")
    
    if os.path.isdir(seg_path):
        print(f"\n--- {specimen} ---")
        
        for file in os.listdir(seg_path):
            if file.endswith(".tif"):
                img = tiff.imread(os.path.join(seg_path, file))
                print(f"{file}: {img.shape}")