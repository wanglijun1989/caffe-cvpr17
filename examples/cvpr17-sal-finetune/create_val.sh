#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

TOOLS=build/tools

DATA_ROOT=/home/lijun/Research/DataSet/Saliency/PASCAL-S/

LIST_PATH=/home/lijun/Research/DataSet/Saliency/PASCAL-S/
#VAL_DATA_ROOT=/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/image/ILSVRC2013_DET_val/
#LMDB_PATH=/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/
LMDB_PATH=/home/lijun/Research/DataSet/Saliency/PASCAL-S/



# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi


#if [ ! -d "$VAL_DATA_ROOT" ]; then
#  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
#  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet validation data is stored."
#  exit 1
#fi

#echo "Creating val img lmdb..."
##
#
##rm $LMDB_PATH/ilsvrc14_train_lmdb -r
#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    $DATA_ROOT \
#    $LIST_PATH/val_sal_img.txt \
#    $LMDB_PATH/img_lmdb
#

echo "Creating val map lmdb..."
rm $LMDB_PATH/map_lmdb -r
#
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --gray=true \
    $DATA_ROOT \
    $LIST_PATH/val_sal_map.txt \
    $LMDB_PATH/map_lmdb



#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    --shuffle \
#    $VAL_DATA_ROOT \
#    $DATA/val.txt \
#    $EXAMPLE/ilsvrc12_val_lmdb
#
echo "Done."
