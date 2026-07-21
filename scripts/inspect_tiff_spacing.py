"""
Inspect a TIFF file's embedded metadata for voxel/pixel spacing info.
Fiji/ImageJ-saved TIFFs often carry this in the XResolution/YResolution
tags and in an ImageJ-specific ImageDescription block (e.g. "spacing=1.2",
"unit=um"), which a plain tifffile.imread() silently discards.

Usage:
    python scripts/inspect_tiff_spacing.py \
        --path /path/to/Dataset_Vitimage2019/CEP011_AS1/CT/registered.tif
"""

import argparse
import tifffile


def main(path):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]

        print(f"File: {path}")
        print(f"Shape: {tf.series[0].shape if tf.series else 'unknown'}")
        print(f"Is ImageJ TIFF: {tf.is_imagej}")
        print()

        # standard TIFF resolution tags
        for tag_name in ("XResolution", "YResolution", "ResolutionUnit"):
            if tag_name in page.tags:
                print(f"{tag_name}: {page.tags[tag_name].value}")

        print()

        # ImageJ-specific metadata (spacing, unit, z-spacing, etc.), stored
        # as a text block, often in ImageDescription
        if tf.imagej_metadata:
            print("ImageJ metadata block:")
            for k, v in tf.imagej_metadata.items():
                print(f"  {k}: {v}")
        else:
            print("No ImageJ metadata block found.")

        # raw ImageDescription, in case it's not parsed as imagej_metadata
        if "ImageDescription" in page.tags:
            print()
            print("Raw ImageDescription tag:")
            print(page.tags["ImageDescription"].value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    main(args.path)