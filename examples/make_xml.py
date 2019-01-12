#coding:utf-8
#author:zhangcc
#date:2019/1/11

import os
import cv2
import csv
import pandas
import shutil
import math
import numpy as np
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import Element,ElementTree

def if_no_exist_path_and_make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
			
	if flag:
		with open(trainval_dataset + "/train.txt", "a") as f:
			f.write(image_name.split(".jpg")[0] + "\n")
			f.close()
		
		with open(trainval_dataset + "/val.txt", "a") as f:
			f.write(image_name.split(".jpg")[0] + "\n")
			f.close

		with open(trainval_dataset + "/trainval.txt", "a") as f:
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
	image_id = 0
	image_name = df['ID'][0]
	image_size = [0, 0, 0]
	for i in range(len(df['ID'])):
		if image_name == df['ID'][i]:
			#print(type(image_name))
			if type(image_name) != str:
				continue
			image = cv2.imread(image_path + "/" + image_name, -1)
			if image is None:
				continue

			image_size = [image.shape[1], image.shape[0], image.shape[2]]
			boxes.append(df['Detection'][i].split(" ", 4))
		else:
			if flag:
				print("train:", image_id, image_name, image_size, len(boxes))
			else:
				print("test:", image_id, image_name, image_size, len(boxes))
			image_id = image_id + 1

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
				continue
			image = cv2.imread(image_path + "/" + image_name, -1)
			if image is None:
				continue

			image_size = [image.shape[1], image.shape[0], image.shape[2]]
			boxes.append(df['Detection'][i].split(" ", 4))

def augment_rotation(txt_input_path, image_input_path, xml_input_path, txt_output_path, image_output_path, xml_output_path):
	for image_name in open(txt_input_path + "/train.txt"):
		image_name = image_name.strip('\n')
		print(image_input_path + "/" + image_name + ".jpg")
		image = cv2.imread(image_input_path + "/" + image_name + ".jpg", -1)
		center_x = image.shape[1] >> 1
		center_y = image.shape[0] >> 1

		for angle in [-180, -90, -6, -4, -2, 0, 2, 4, 6, 90]:
			print("angle: ", angle)
			M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
			image_RT = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
			image_mark = image_RT.copy()
			cv2.imwrite(image_output_path + "/" + image_name + '_' + str(angle) + '_rotate.jpg', image_RT)

			shutil.copyfile(xml_input_path + "/" + image_name + ".xml", xml_output_path + "/" + image_name + '_' + str(angle) + "_rotate.xml")

			print(xml_output_path + "/" + image_name + '_' + str(angle) + "_rotate.xml")
			tree = ET.parse(xml_output_path + "/" + image_name + '_' + str(angle) + "_rotate.xml")
			root = tree.getroot()
			root.find('filename').text = image_name + '_' + str(angle) + '_rotate.jpg'

			for obj in root.findall('object'):
				def rotate_point(x0, y0, center_x, center_y, theta):
					pi = 3.1415926535897932384626433832795
					theta = - theta * pi / 180.0
					x = (x0 - center_x) * math.cos(theta) - (y0 - center_y) * math.sin(theta) + center_x
					y = (x0 - center_x) * math.sin(theta) + (y0 - center_y) * math.cos(theta) + center_y
					
					x = math.floor(x)
					y = math.floor(y)

					if x >= (image.shape[1] - 1): x = image.shape[1] - 1
					elif x < 0: x = 0
					
					if y >= (image.shape[0] - 1): y = image.shape[0] - 1
					elif y < 0: y = 0

					return x, y

				shape_color_type = obj.find('shape').get('color')
				color_type = (0, 0, 255)
				if 'Red' == shape_color_type:
					color_type = (0, 0, 255)
				elif 'Green' == shape_color_type:
					color_type = (0, 255, 0)

				xmin = int(int(obj.find('bndbox/xmin').text) / 1)
				ymin = int(int(obj.find('bndbox/ymin').text) / 1)
				xmax = int(int(obj.find('bndbox/xmax').text) / 1)
				ymax = int(int(obj.find('bndbox/ymax').text) / 1)

				x00 = xmin
				y00 = ymin

				x11 = xmax
				y11 = ymax

				x01 = x11
				y01 = y00

				x10 = x00
				y10 = y11

				t_x00, t_y00 = rotate_point(x00, y00, center_x, center_y, angle)
				t_x01, t_y01 = rotate_point(x01, y01, center_x, center_y, angle)
				t_x10, t_y10 = rotate_point(x10, y10, center_x, center_y, angle)
				t_x11, t_y11 = rotate_point(x11, y11, center_x, center_y, angle)

				t_xarray = [t_x00, t_x01, t_x10, t_x11]
				t_yarray = [t_y00, t_y01, t_y10, t_y11]
				
				t_xmin = min(t_xarray)
				t_ymin = min(t_yarray)
				t_xmax = max(t_xarray)
				t_ymax = max(t_yarray)

				obj.find('bndbox/xmin').text = str(t_xmin)
				obj.find('bndbox/ymin').text = str(t_ymin)
				obj.find('bndbox/xmax').text = str(t_xmax)
				obj.find('bndbox/ymax').text = str(t_ymax)

				obj.findall('shape/points/x')[0].text = obj.find('bndbox/xmin').text
				obj.findall('shape/points/y')[0].text = obj.find('bndbox/ymin').text
				obj.findall('shape/points/x')[1].text = obj.find('bndbox/xmax').text
				obj.findall('shape/points/y')[1].text = obj.find('bndbox/ymax').text

				tree.write(xml_output_path + "/" + image_name + '_' + str(angle) + "_rotate.xml", encoding='utf-8', xml_declaration=True)
				cv2.rectangle(image_mark, (t_xmin, t_ymin), (t_xmax, t_ymax), color_type, 5)
				cv2.imwrite(image_output_path + "/" + image_name + '_' + str(angle)  + "_rotate_MARKED.jpg", image_mark)
				if 0:
					#cv2.rectangle(image_mark, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
					#cv2.rectangle(image_mark, (t_xmin, t_ymin), (t_xmax, t_ymax), (0, 255, 0), 5)
					#cv2.circle(image_mark, (t_xmin, t_ymin), 5, (0, 255, 0), -1)
					#cv2.circle(image_mark, (t_xmax, t_ymax), 5, (0, 255, 0), -1)
					#cv2.circle(image_mark, (t_x00, t_y00), 5, (255, 0, 0), -1)
					#cv2.circle(image_mark, (t_x01, t_y01), 5, (0, 255, 0), -1)
					#cv2.circle(image_mark, (t_x10, t_y10), 5, (0, 255, 0), -1)
					#cv2.circle(image_mark, (t_x11, t_y11), 5, (0, 255, 0), -1)				
					cv2.namedWindow(str(angle), 0)
					cv2.imshow(str(angle), image_mark)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
					break
				
def csvToxml():
	train_csv_file_name = "train_labels.csv"
	test_csv_file_name = "submit_example.csv"
	
	image_path = "data/VOCdevkit2007/VOC2007/JPEGImages"
	xml_output_path = "data/VOCdevkit2007/VOC2007/Annotations"
	trainval_dataset = "data/VOCdevkit2007/VOC2007/ImageSets/Main"
	
	parse_csv(train_csv_file_name, image_path, xml_output_path, trainval_dataset, 1)
	#parse_csv(test_csv_file_name, image_path, xml_output_path, trainval_dataset, 0)

def augment():
	txt_input_path = "data/VOCdevkit2007/VOC2007/ImageSets/Main"
	image_input_path = "data/VOCdevkit2007/VOC2007/JPEGImages"
	xml_input_path = "data/VOCdevkit2007/VOC2007/Annotations"

	txt_output_path = "data/VOCdevkit2007/VOC2007_rotation/ImageSets/Main"
	image_output_path = "data/VOCdevkit2007/VOC2007_rotation/JPEGImages"
	xml_output_path = "data/VOCdevkit2007/VOC2007_rotation/Annotations"
	
	if_no_exist_path_and_make_path(txt_output_path)
	if_no_exist_path_and_make_path(image_output_path)
	if_no_exist_path_and_make_path(xml_output_path)

	augment_rotation(txt_input_path, image_input_path, xml_input_path, \
					txt_output_path, image_output_path, xml_output_path)

if __name__ == "__main__":
	#csvToxml()
	augment()

