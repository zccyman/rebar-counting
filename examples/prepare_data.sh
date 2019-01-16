#!/bin/sh

TARGET_DIR=data/VOCdevkit2007/VOC2007

rm -rf $TARGET_DIR

mkdir -p $TARGET_DIR

cp -rf data/VOCdevkit2007/VOC2007_origin/* $TARGET_DIR

rotation_array=(-180 -90 -6 -4 -2 0 2 4 6 90 180) 
for k in ${rotation_array[@]}
do 
cp -rf data/VOCdevkit2007/VOC2007_rotation_${k}/* $TARGET_DIR
done

flip_array=(-1 0 1) 
for k in ${flip_array[@]}
do 
cp -rf data/VOCdevkit2007/VOC2007_flip_${k}/* $TARGET_DIR
done

crop_array=(0 1 2) 
for k in ${crop_array[@]}
do 
cp -rf data/VOCdevkit2007/VOC2007_crop_${k}/* $TARGET_DIR
done

rm -rf data/VOCdevkit2007/VOC2007/ImageSets/Main/*.txt

echo "All done"
