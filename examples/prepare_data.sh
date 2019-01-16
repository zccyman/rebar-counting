#!/bin/sh

TARGET_DIR=data/VOCdevkit2007/VOC2007

rm -rf $TARGET_DIR

mkdir -p $TARGET_DIR

cp -rf data/VOCdevkit2007/VOC2007_origin/* $TARGET_DIR

rotation_array=(-180 -90 -6 -4 -2 0 2 4 6 90) 
for ((k=0;k<${rotation_array[@]};k++))
do 
cp -rf data/VOCdevkit2007/VOC2007_rotation_${rotation_array[k]}/* $TARGET_DIR
done

flip_array=(-1 0 1) 
for ((k=0;k<${flip_array[@]};k++))
do 
cp -rf data/VOCdevkit2007/VOC2007_rotation_${flip_array[k]}/* $TARGET_DIR
done

crop_array=(0) 
for ((k=0;k<${crop_array[@]};k++))
do 
cp -rf data/VOCdevkit2007/VOC2007_crop_${crop_array[k]}/* $TARGET_DIR
done

rm -rf data/VOCdevkit2007/VOC2007/ImageSets/Main/*.txt

echo "All done"
