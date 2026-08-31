#!/bin/bash
# DRR generation with CORRECTED geometry: --detector-size 3072 --gpu,
# real 0.139mm pixel spacing now actually written into output metadata
# (calibrate_drr.py fix applied). Flat output:
# results/PXR_DRR_2026_final/CEP_<specimen>.tif
#
# IMPORTANT: the OFFSET values below are the OLD ones, tuned against
# the broken pre-fix geometry (318 already confirmed needing a
# different offset once geometry was corrected). Treat every one of
# these as an unverified starting guess, not a confirmed value,
# spot-check each specimen visually before trusting the batch.

set -e

XR_DIR="/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/code_python/VineRadiologistAI/dataset/ct_volumes_table_removed"
PXR_DIR="dataset/radiograph_portable_2026_tif"
DRR_OUT="results/PXR_DRR_2026_final"
mkdir -p "$DRR_OUT"

declare -A OFFSET=(
  [313B]=-140 [318]=-90  [322]=-70  [323]=-60  [330]=-40
  [335]=-60   [368B]=-40 [378A]=-10 [378B]=-30 [380A]=-20
  [764B]=-40  [988B]=-100 [1181]=-90 [1186A]=-40 [1189]=-70
  [1191]=-20  [1193]=-90 [1195]=-20 [1266A]=-90 [2184A]=-50
)

declare -A PXR_FILE=(
  [313B]=69ca5d9ee42f5e088ff0f93b.tif
  [318]=69ca5db1e42f5e09df2bb4a6.tif
  [322]=69ca5dc4e42f5e09df2bb4ac.tif
  [323]=69ca5de6e42f5e09b4512919.tif
  [330]=69ca5df7e42f5e09df2bb4b0.tif
  [335]=69ca5e1be42f5e0a22520ec4.tif
  [368B]=69ca536b2ad9c81239913207.tif
  [378A]=69ca4b43e42f5e026f6dd0e7.tif
  [378B]=69ca5613e42f5e07eeeed92b.tif
  [380A]=69ca562de42f5e0682a2cf2d.tif
  [764B]=69ca5d80e42f5e088ff0f937.tif
  [988B]=69ca5645e42f5e0682a2cf31.tif
  [1181]=69ca6a532ad9c81239913234.tif
  [1186A]=69ca569de42f5e071ae9bcf3.tif
  [1189]=69ca67ad2ad9c8123991322c.tif
  [1191]=69ca5e44e42f5e0a22520ec8.tif
  [1193]=69ca5e6fe42f5e09df2bb4b4.tif
  [1195]=69ca56b6e42f5e071ae9bcfa.tif
  [1266A]=69ca56d2e42f5e0702f748da.tif
  [2184A]=69ca5676e42f5e071ae9bced.tif
)

declare -A SPACING=(
  [313B]=0.578125 [318]=0.562500  [322]=0.851563  [323]=0.585937  [330]=0.697265
  [335]=0.820312  [368B]=0.628906 [378A]=0.617187 [378B]=0.681640 [380A]=0.828125
  [764B]=0.751953 [988B]=0.623047 [1181]=0.585937 [1186A]=0.605469 [1189]=0.683593
  [1191]=0.582031 [1193]=0.730469 [1195]=0.867187 [1266A]=0.615234 [2184A]=0.630859
)

declare -A SPACING_Z=(
  [313B]=0.4 [318]=0.4  [322]=0.4  [323]=0.6  [330]=0.4
  [335]=0.4  [368B]=0.4 [378A]=0.4 [378B]=0.4 [380A]=0.4
  [764B]=0.4 [988B]=0.4 [1181]=0.7 [1186A]=0.4 [1189]=0.4
  [1191]=0.4 [1193]=0.4 [1195]=0.4 [1266A]=0.4 [2184A]=0.4
)

for specimen in "${!OFFSET[@]}"; do
  echo "=== $specimen ==="
  python scripts/calibrate_drr.py \
    --xr "$XR_DIR/CEP_${specimen}_2026_XR.tif" \
    --pxr "$PXR_DIR/CEP_${specimen}/${PXR_FILE[$specimen]}" \
    --voxel-spacing-mm "${SPACING[$specimen]}" \
    --voxel-spacing-z-mm "${SPACING_Z[$specimen]}" \
    --sid-mm 1230 --spd-mm 800 \
    --offset-v-mm "${OFFSET[$specimen]}" \
    --detector-size 3072 --gpu --row-chunk 16 \
    --metric wasserstein \
    --fixed-attenuation 0.02 \
    --out-drr "$DRR_OUT/CEP_${specimen}.tif"
done#!/bin/bash
# DRR generation with CORRECTED geometry: --detector-size 3072 --gpu,
# real 0.139mm pixel spacing now actually written into output metadata
# (calibrate_drr.py fix applied). Flat output:
# results/PXR_DRR_2026_final/CEP_<specimen>.tif
#
# IMPORTANT: the OFFSET values below are the OLD ones, tuned against
# the broken pre-fix geometry (318 already confirmed needing a
# different offset once geometry was corrected). Treat every one of
# these as an unverified starting guess, not a confirmed value,
# spot-check each specimen visually before trusting the batch.

set -e

XR_DIR="/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/code_python/VineRadiologistAI/dataset/ct_volumes_table_removed"
PXR_DIR="dataset/radiograph_portable_2026_tif"
DRR_OUT="results/PXR_DRR_2026_final"
mkdir -p "$DRR_OUT"

declare -A OFFSET=(
  [313B]=-140 [318]=-90  [322]=-70  [323]=-60  [330]=-40
  [335]=-60   [368B]=-40 [378A]=-10 [378B]=-30 [380A]=-20
  [764B]=-40  [988B]=-100 [1181]=-90 [1186A]=-40 [1189]=-70
  [1191]=-20  [1193]=-90 [1195]=-20 [1266A]=-90 [2184A]=-50
)

declare -A PXR_FILE=(
  [313B]=69ca5d9ee42f5e088ff0f93b.tif
  [318]=69ca5db1e42f5e09df2bb4a6.tif
  [322]=69ca5dc4e42f5e09df2bb4ac.tif
  [323]=69ca5de6e42f5e09b4512919.tif
  [330]=69ca5df7e42f5e09df2bb4b0.tif
  [335]=69ca5e1be42f5e0a22520ec4.tif
  [368B]=69ca536b2ad9c81239913207.tif
  [378A]=69ca4b43e42f5e026f6dd0e7.tif
  [378B]=69ca5613e42f5e07eeeed92b.tif
  [380A]=69ca562de42f5e0682a2cf2d.tif
  [764B]=69ca5d80e42f5e088ff0f937.tif
  [988B]=69ca5645e42f5e0682a2cf31.tif
  [1181]=69ca6a532ad9c81239913234.tif
  [1186A]=69ca569de42f5e071ae9bcf3.tif
  [1189]=69ca67ad2ad9c8123991322c.tif
  [1191]=69ca5e44e42f5e0a22520ec8.tif
  [1193]=69ca5e6fe42f5e09df2bb4b4.tif
  [1195]=69ca56b6e42f5e071ae9bcfa.tif
  [1266A]=69ca56d2e42f5e0702f748da.tif
  [2184A]=69ca5676e42f5e071ae9bced.tif
)

declare -A SPACING=(
  [313B]=0.578125 [318]=0.562500  [322]=0.851563  [323]=0.585937  [330]=0.697265
  [335]=0.820312  [368B]=0.628906 [378A]=0.617187 [378B]=0.681640 [380A]=0.828125
  [764B]=0.751953 [988B]=0.623047 [1181]=0.585937 [1186A]=0.605469 [1189]=0.683593
  [1191]=0.582031 [1193]=0.730469 [1195]=0.867187 [1266A]=0.615234 [2184A]=0.630859
)

declare -A SPACING_Z=(
  [313B]=0.4 [318]=0.4  [322]=0.4  [323]=0.6  [330]=0.4
  [335]=0.4  [368B]=0.4 [378A]=0.4 [378B]=0.4 [380A]=0.4
  [764B]=0.4 [988B]=0.4 [1181]=0.7 [1186A]=0.4 [1189]=0.4
  [1191]=0.4 [1193]=0.4 [1195]=0.4 [1266A]=0.4 [2184A]=0.4
)

for specimen in "${!OFFSET[@]}"; do
  echo "=== $specimen ==="
  python scripts/calibrate_drr.py \
    --xr "$XR_DIR/CEP_${specimen}_2026_XR.tif" \
    --pxr "$PXR_DIR/CEP_${specimen}/${PXR_FILE[$specimen]}" \
    --voxel-spacing-mm "${SPACING[$specimen]}" \
    --voxel-spacing-z-mm "${SPACING_Z[$specimen]}" \
    --sid-mm 1230 --spd-mm 800 \
    --offset-v-mm "${OFFSET[$specimen]}" \
    --detector-size 3072 --gpu --row-chunk 16 \
    --metric wasserstein \
    --fixed-attenuation 0.02 \
    --out-drr "$DRR_OUT/CEP_${specimen}.tif"
done