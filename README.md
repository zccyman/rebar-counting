# rebar-counting


## Prepare Data
	
- data/VOCdevkit/
	- Annotations
	- ImageSets
	- JPEGImages
	
download data from [here](https://pan.baidu.com/s/1NxACM1coAXthmXizXKyhow?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=%2Fgithub%2Fpublic%2Frebar-counting&parentPath=%2Fgithub/VOCdevkit.zip)

## Create Datasets Using VOC Formate
```
python make_xml.py
```