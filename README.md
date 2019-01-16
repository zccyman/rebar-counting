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
	- Rotation: [-180, -90, -6, -4, -2, 0, 2, 4, 6, 90]
	```
	sh data_augment_rotate.sh {-180, -90, -6, -4, -2, 0, 2, 4, 6, 90}
	```
	- Flip: [-1, 0, 1]
	```
	sh data_augment_flip.sh {-1, 0, 1}
	```
	- Crop: [1, 2, 3,..., n]
	```
	sh data_augment_crop.sh {1, 2, 3}
	```