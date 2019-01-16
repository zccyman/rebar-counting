# rebar-counting


## Prepare Data
	
- data/VOCdevkit2007/VOC2007_origin
	- Annotations
	- ImageSets
	- JPEGImages
	
download data from [here](https://pan.baidu.com/s/1NxACM1coAXthmXizXKyhow?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=%2Fgithub%2Fpublic%2Frebar-counting&parentPath=%2Fgithub/VOCdevkit.zip)

## Create Datasets Using VOC Formate
```
python make_xml.py
```

## Data Augment
- Methods
	- Rotation
	```
	sh data_augment_rotate.sh 2
	```
	- Flip
	```
	sh data_augment_flip.sh 1
	```
	- Crop
	```
	sh data_augment_crop.sh 0
	```