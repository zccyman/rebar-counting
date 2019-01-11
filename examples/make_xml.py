#coding:utf-8
#author:zhangcc
#date:2019/1/11

import os
import cv2
import csv
import pandas
import numpy as np
import xml.etree.cElementTree as ET

def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
			
def create_xml(image_name, image_size, boxes, xml_output_path):
	xml_name = image_name.split(".jpg")[0]
	
	root = ET.Element("annotation")

	ET.SubElement(root, "CreateVersion").text = str(2.5)
	ET.SubElement(root, "folder").text = " "

	#image_name
	ET.SubElement(root, "filename").text = image_name
	ET.SubElement(root, "path").text = " "

	source_node = ET.SubElement(root, "source")
	ET.SubElement(source_node, "database").text = "Unknown"

	ET.SubElement(root, "score").text = str(0)

	#image_size
	size_node = ET.SubElement(root, "size")
	
	ET.SubElement(size_node, "width").text = str(image_size[0])
	ET.SubElement(size_node, "height").text = str(image_size[1])
	ET.SubElement(size_node, "depth").text = str(image_size[2])
	
	ET.SubElement(root, "segmented").text = str(0)

	for i in range(len(boxes)):
		#add object
		object = ET.SubElement(root, "object")
		ET.SubElement(object, "name").text = "rebar"
		ET.SubElement(object, "pose").text = str(0)
		ET.SubElement(object, "truncated").text = str(0)
		ET.SubElement(object, "difficult").text = str(0)
		
		object_bndbox_node = ET.SubElement(object, "bndbox")
		ET.SubElement(object_bndbox_node, "xmin").text = str(boxes[i][0])
		ET.SubElement(object_bndbox_node, "ymin").text = str(boxes[i][1])
		ET.SubElement(object_bndbox_node, "xmax").text = str(boxes[i][2])
		ET.SubElement(object_bndbox_node, "ymax").text = str(boxes[i][3])

		object_shape_node = ET.SubElement(object, "shape")
		shape_points_node = ET.SubElement(object_shape_node, "points")
		shape_points_node.attrib = {"type": "rect", "color": "Green", "thickness": "1"}
		
		ET.SubElement(shape_points_node, "x").text = str(boxes[i][0])
		ET.SubElement(shape_points_node, "y").text = str(boxes[i][1])
		ET.SubElement(shape_points_node, "x").text = str(boxes[i][2])
		ET.SubElement(shape_points_node, "y").text = str(boxes[i][3])
	
	tree = ET.ElementTree(root)

	indent(root)

	tree.write(xml_output_path + "/" + xml_name + ".xml")

def create_txt(image_name, trainval_dataset, flag):
	if not os.path.exists(trainval_dataset):
		os.makedirs(trainval_dataset)

	with open(trainval_dataset + "/trainval.txt", "a") as f:
			f.write(image_name.split(".jpg")[0] + "\n")
			f.close()
			
	if flag:
		with open(trainval_dataset + "/train.txt", "a") as f:
			f.write(image_name.split(".jpg")[0] + "\n")
			f.close()
	else:
		with open(trainval_dataset + "/test.txt", "a") as f:
			f.write(image_name.split(".jpg")[0] + "\n")
			f.close()

def parse_csv(csv_file_name, image_path, xml_output_path, trainval_dataset, flag):
	df = pandas.read_csv(csv_file_name)
	#print(df)
	#print(df['ID'])
	#print(df['Detection'][0])
	#x = df['Detection'][0].split(" ", 1)
	#print(type(x))
	#input("")
	
	boxes = []
	image_name = df['ID'][0]
	for i in range(len(df['ID'])):
		if image_name == df['ID'][i]:
			#print(type(image_name))
			if type(image_name) != str:
				image_size = [0, 0, 0]
				continue
			image = cv2.imread(image_path + "/" + image_name, -1)
			if image is None:
				image_size = [0, 0, 0]
				continue

			image_size = [image.shape[1], image.shape[0], image.shape[2]]
			boxes.append(df['Detection'][i].split(" ", 4))
		else:
			print(image_name, image_size, len(boxes))
			#print(boxes)
			#input("")
			if 0:
				for i in range(len(boxes)):
					#print(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], type(boxes[i][0]))
					x0 = int(boxes[i][0])
					y0 = int(boxes[i][1])
					x1 = int(boxes[i][2])
					y1 = int(boxes[i][3])
					cv2.rectangle(image, (x0, y0), (x1,y1), (1, 1,255), 1)
				cv2.namedWindow(image_name, 0)
				cv2.imshow(image_name, image)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

			create_xml(image_name, image_size, boxes, xml_output_path)
			create_txt(image_name, trainval_dataset, flag)
			
			boxes = []
			image_name = df['ID'][i]
			#print(type(image_name))
			if type(image_name) != str:
				image_size = [0, 0, 0]
				continue
			image = cv2.imread(image_path + "/" + image_name, -1)
			if image is None:
				image_size = [0, 0, 0]
				continue

			image_size = [image.shape[1], image.shape[0], image.shape[2]]
			boxes.append(df['Detection'][i].split(" ", 4))
	
if __name__ == "__main__":
	train_csv_file_name = "train_labels.csv"
	test_csv_file_name = "submit_example.csv"
	
	image_path = "data/VOCdevkit2007/VOC2007/JPEGImages"
	xml_output_path = "data/VOCdevkit2007/VOC2007/Annotations"
	trainval_dataset = "data/VOCdevkit2007/VOC2007/ImageSets/Main"
	
	parse_csv(train_csv_file_name, image_path, xml_output_path, trainval_dataset, 1)
	parse_csv(test_csv_file_name, image_path, xml_output_path, trainval_dataset, 0)
