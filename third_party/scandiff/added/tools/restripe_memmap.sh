#!/bin/bash
# Rewrite a feature memmap with Lustre striping across all OSTs.
#
# WHY: the memmaps were written with stripe_count=1, i.e. each ~200-400 GB file
# lives on a SINGLE OST. Measured consequence: random 2 MB reads saturate at
# ~124 MB/s and do NOT scale with reader threads (1, 4 and 12 threads all give
# the same throughput), because every dataloader worker queues on the same
# storage target. Sequential read from the same filesystem reaches 1.0 GB/s.
# With stripe_count=8 the reads spread over all 8 OSTs.
#
# Striping cannot be changed in place -- the file must be rewritten with the
# new layout, so this needs free space equal to the file size.
#
#   bash tools/restripe_memmap.sh memmap_oursynth_200k
set -eo pipefail
WS=/mnt/lustre-rzg/workspaces/ws/nim00018/u27846-nim18-dinov2-features
NAME="${1:?usage: restripe_memmap.sh <memmap_dir_name>}"
SRC="$WS/$NAME"
[ -d "$SRC" ] || { echo "no such memmap: $SRC" >&2; exit 1; }

SIZE=$(du -sb "$SRC/features.npy" | cut -f1)
AVAIL=$(df -B1 --output=avail "$WS" | tail -1)
echo "$NAME: features.npy $(numfmt --to=iec $SIZE), free $(numfmt --to=iec $AVAIL)"
[ "$AVAIL" -gt "$SIZE" ] || { echo "not enough free space" >&2; exit 1; }

echo "current stripe_count: $(lfs getstripe -c "$SRC/features.npy")"
TMP="$SRC/features.npy.striped"
rm -f "$TMP"
lfs setstripe -c 8 -S 1M "$TMP"          # layout must exist BEFORE writing
echo "copying with stripe_count=8 ..."
time dd if="$SRC/features.npy" of="$TMP" bs=16M status=progress

# verify size, then swap
[ "$(stat -c%s "$TMP")" = "$SIZE" ] || { echo "size mismatch, keeping original" >&2; exit 1; }
mv "$SRC/features.npy" "$SRC/features.npy.old"
mv "$TMP" "$SRC/features.npy"
echo "new stripe_count: $(lfs getstripe -c "$SRC/features.npy")"
echo "old file kept as features.npy.old -- delete after a successful training run:"
echo "  rm $SRC/features.npy.old"
