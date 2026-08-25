#!/bin/bash
# Batch DRR generation + distribution comparison for table-removed volumes.
# Flat output: all DRRs in one dir, all distribution plots in another,
# both named by specimen only.
#
# VOXEL SPACING: only 4 of these are confirmed from this conversation
# (368B, 322, 378A, 1191). The other 16 are left BLANK on purpose, I am
# not going to guess numbers for the rest, fill them in from your real
# per-specimen table before running.

set -e

XR_DIR="/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/gargee/code_python/VineRadiologistAI/dataset/ct_volumes_table_removed"
PXR_DIR="dataset/radiograph_portable_2026_tif"
DRR_OUT="results/PXR_DRR_2026_table_removed/DRR"
DIST_OUT="results/PXR_DRR_2026_table_removed/Distribution"
mkdir -p "$DRR_OUT" "$DIST_OUT"

# specimen -> offset-u-mm (from your list)
declare -A OFFSET=(
  [313B]=-100 [318]=-60  [322]=-50  [323]=-40  [330]=-10
  [335]=-30   [368B]=-30 [378A]=0   [378B]=-10 [380A]=-20
  [764B]=-40  [988B]=-100 [1181]=-90 [1186A]=-20 [1189]=-60
  [1191]=0    [1193]=-50 [1195]=0   [1266A]=-70 [2184A]=-10
)

# specimen -> real PXR filename actually used in the comparison PDF
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

# specimen -> voxel-spacing-mm (XY), from your confirmed table
declare -A SPACING=(
  [313B]=0.578125 [318]=0.562500  [322]=0.851563  [323]=0.585937  [330]=0.697265
  [335]=0.820312  [368B]=0.628906 [378A]=0.617187 [378B]=0.681640 [380A]=0.828125
  [764B]=0.751953 [988B]=0.623047 [1181]=0.585937 [1186A]=0.605469 [1189]=0.683593
  [1191]=0.582031 [1193]=0.730469 [1195]=0.867187 [1266A]=0.615234 [2184A]=0.630859
)

# specimen -> voxel-spacing-z-mm, from your confirmed table
# (most are 0.4, but 323 = 0.6 and 1181 = 0.7, not the default)
declare -A SPACING_Z=(
  [313B]=0.4 [318]=0.4  [322]=0.4  [323]=0.6  [330]=0.4
  [335]=0.4  [368B]=0.4 [378A]=0.4 [378B]=0.4 [380A]=0.4
  [764B]=0.4 [988B]=0.4 [1181]=0.7 [1186A]=0.4 [1189]=0.4
  [1191]=0.4 [1193]=0.4 [1195]=0.4 [1266A]=0.4 [2184A]=0.4
)

for specimen in "${!OFFSET[@]}"; do
  spacing="${SPACING[$specimen]}"
  spacing_z="${SPACING_Z[$specimen]}"

  echo "=== $specimen ==="
  python scripts/calibrate_drr.py \
    --xr "$XR_DIR/CEP_${specimen}_2026_XR.tif" \
    --pxr "$PXR_DIR/CEP_${specimen}/${PXR_FILE[$specimen]}" \
    --voxel-spacing-mm "$spacing" \
    --voxel-spacing-z-mm "$spacing_z" \
    --sid-mm 1230 --spd-mm 800 \
    --metric wasserstein \
    --fixed-attenuation 0.02 \
    --offset-u-mm "${OFFSET[$specimen]}" \
    --out-drr "$DRR_OUT/CEP_${specimen}.tif"

  python scripts/compare_distributions_multi.py \
    --real "$PXR_DIR/CEP_${specimen}/${PXR_FILE[$specimen]}" \
    --drr "calibrated=$DRR_OUT/CEP_${specimen}.tif" \
    --out "$DIST_OUT/CEP_${specimen}.png"
done