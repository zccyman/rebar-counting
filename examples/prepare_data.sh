#!/bin/sh

rm -rf data/VOCdevkit2007/VOC2007

cp -rf data/VOCdevkit2007/VOC2007_origin data/VOCdevkit2007/VOC2007
cp -rf data/VOCdevkit2007/VOC2007_rotation data/VOCdevkit2007/VOC2007
cp -rf data/VOCdevkit2007/VOC2007_flip data/VOCdevkit2007/VOC2007

rm -rf data/VOCdevkit2007/VOC2007/ImageSets/Main/*.txt
