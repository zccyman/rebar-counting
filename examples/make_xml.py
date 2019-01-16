#coding:utf-8
#author:zhangcc
#date:2019/1/11

import os
import sys
import argparse
import cv2
import csv
import random
import pandas
import shutil
import math
import numpy as np
from numba import jit
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import Element,ElementTree

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Data Augment for rebar counting')
parser.add_argument('--is_rotate', dest='is_rotate', help='whether to rotate image for augment', default=0, type=int)
parser.add_argument('--rotation_angle', dest='rotation_angle', help='rotation_angle', default=180, type=int)
parser.add_argument('--is_flip', dest='is_flip', help='whether to flip image for augment', default=0, type=int)
parser.add_argument('--flip_type', dest='flip_type', help='the method of flip', default=0, type=int)
parser.add_argument('--is_crop', dest='is_crop', help='whether to crop image for augment', default=0, type=int)
parser.add_argument('--crop_num', dest='crop_num', help='number of crop', default=0, type=int)


args = parser.parse_args()

ratio = []
box_w = []
box_h = []

#@jit
def if_no_exist_path_and_make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

#@jit
def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)+1
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()

#@jit
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

#@jit			
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
		ET.SubElement(object, "staintype").text = " "
		ET.SubElement(object, "level").text = str(1)
	
		if 0:
			ratio_hw = (int(boxes[i][3]) - int(boxes[i][1])) / (int(boxes[i][2]) - int(boxes[i][0]))
			box_h_ = (int(boxes[i][3]) - int(boxes[i][1])) / image_size[1]
			box_w_ = (int(boxes[i][2]) - int(boxes[i][0])) / image_size[0]
		
			ratio.append(ratio_hw)
			box_h.append(box_h_)
			box_w.append(box_w_)

			#show and saveimage
			fig0 = plt.figure(0)
			plt.cla()
			plt.scatter(range(len(ratio)), ratio, marker = 'o', color = 'r', s = 15)
			fig0.savefig('doc/images/ratio.jpg')

			fig1 = plt.figure(1)
			plt.cla()
			plt.scatter(range(len(box_h)), box_h, marker = 'o', color = 'r', s = 15)
			fig1.savefig('doc/images/box_h.jpg')

			fig2 = plt.figure(2)
			plt.cla()
			plt.scatter(range(len(box_w)), box_w, marker = 'o', color = 'r', s = 15)
			fig2.savefig('doc/images/box_w.jpg')
		
		object_bndbox_node = ET.SubElement(object, "bndbox")
		ET.SubElement(object_bndbox_node, "xmin").text = str(boxes[i][0])
		ET.SubElement(object_bndbox_node, "ymin").text = str(boxes[i][1])
		ET.SubElement(object_bndbox_node, "xmax").text = str(boxes[i][2])
		ET.SubElement(object_bndbox_node, "ymax").text = str(boxes[i][3])

		object_shape_node = ET.SubElement(object, "shape")
		shape_points_node = ET.SubElement(object_shape_node, "points")
		shape_points_node.attrib = {"type":"rect","color":"Red","thickness":"3"}
		
		ET.SubElement(shape_points_node, "x").text = str(boxes[i][0])
		ET.SubElement(shape_points_node, "y").text = str(boxes[i][1])
		ET.SubElement(shape_points_node, "x").text = str(boxes[i][2])
		ET.SubElement(shape_points_node, "y").text = str(boxes[i][3])
	
	tree = ET.ElementTree(root)

	indent(root)

	tree.write(xml_output_path + "/" + xml_name + ".xml")

#@jit
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

#@jit
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
			#if type(image_name) != str:
			if image_name == "end":
				print("image_name is not exist...")
				continue
			image = cv2.imread(image_path + "/" + image_name, -1)
			if image is None:
				print("image is NULL...")
				continue

			image_size = [image.shape[1], image.shape[0], image.shape[2]]
			boxes.append(df['Detection'][i].split(" ", 4))
		else:
			if 0 == image_id:
				for _str in ["test.txt", "train.txt", "val.txt", "trainval.txt"]:
					t_path = trainval_dataset + "/" + _str
					if os.path.exists(t_path):
						os.remove(t_path)

			if flag:
				print("train:", image_id, image_name, image_size, len(boxes))
			else:
				print("test:", image_id, image_name, image_size, len(boxes))
			image_id = image_id + 1

			#print(boxes)
			#input("")
			
			image_mark = image.copy()
			for box_idx in range(len(boxes)):
				#print(boxes[box_idx][0], boxes[box_idx][1], boxes[box_idx][2], boxes[box_idx][3], type(boxes[i][0]))
				x0 = int(boxes[box_idx][0])
				y0 = int(boxes[box_idx][1])
				x1 = int(boxes[box_idx][2])
				y1 = int(boxes[box_idx][3])
				cv2.rectangle(image_mark, (x0, y0), (x1,y1), (1, 1, 255), 5)
				cv2.circle(image_mark, (x0, y0), 5, (255, 1, 1), -1) #Blue
				cv2.circle(image_mark, (x1,y1), 5, (1, 255, 1), -1) #Green
				cv2.imwrite(image_path + "/" + image_name.split(".jpg")[0] + "_MARKED.jpg", image_mark)
			if 0:
				cv2.namedWindow(image_name, 0)
				cv2.imshow(image_name, image)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			
			create_xml(image_name, image_size, boxes, xml_output_path)
			create_txt(image_name, trainval_dataset, flag)
			
			boxes = []
			image_name = df['ID'][i]
			#print(type(image_name))
			#if type(image_name) != str:
			if image_name == "end":
				print("image_name is not exist...")
				continue
			image = cv2.imread(image_path + "/" + image_name, -1)
			if image is None:
				print("image is NULL...")
				continue

			image_size = [image.shape[1], image.shape[0], image.shape[2]]
			boxes.append(df['Detection'][i].split(" ", 4))

#@jit
def augment_rotation(angle_idx, angle_array, txt_input_path, image_input_path, xml_input_path, txt_output_path, image_output_path, xml_output_path):
	angle_sin_array = []
	angle_cos_array = []
	for theta_ in angle_array:
		angle_sin_array.append(math.sin(theta_ * (np.pi / 180.0)))
		angle_cos_array.append(math.cos(theta_ * (np.pi / 180.0)))

	total_images = len(open(txt_input_path + "/train.txt",'rU').readlines())

	image_id = 0
	for image_name in open(txt_input_path + "/train.txt"):
		image_name = image_name.strip('\n')
		print(image_input_path + "/" + image_name + ".jpg")
		image = cv2.imread(image_input_path + "/" + image_name + ".jpg", -1)
		if image is None:
			print("image is NULL...")
			continue

		center_x = image.shape[1] >> 1
		center_y = image.shape[0] >> 1

		view_bar(image_id, total_images)
		image_id = image_id + 1

		M = cv2.getRotationMatrix2D((center_x, center_y), angle_array[angle_idx], 1.0)
		image_RT = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
		image_mark = image_RT.copy()
		cv2.imwrite(image_output_path + "/" + image_name + '_' + str(angle_array[angle_idx]) + '_rotate.jpg', image_RT)

		shutil.copyfile(xml_input_path + "/" + image_name + ".xml", xml_output_path + "/" + image_name + '_' + str(angle_array[angle_idx]) + "_rotate.xml")

		print(xml_output_path + "/" + image_name + '_' + str(angle_array[angle_idx]) + "_rotate.xml")
		tree = ET.parse(xml_output_path + "/" + image_name + '_' + str(angle_array[angle_idx]) + "_rotate.xml")
		root = tree.getroot()
		root.find('filename').text = image_name + '_' + str(angle_array[angle_idx]) + '_rotate.jpg'

		for obj in root.findall('object'):
			def rotate_point(x0, y0, center_x, center_y, theta, idx):
				#pi = 3.1415926535897932384626433832795
				#theta = - theta * pi / 180.0
				#x = (x0 - center_x) * math.cos(theta) - (y0 - center_y) * math.sin(theta) + center_x
				#y = (x0 - center_x) * math.sin(theta) + (y0 - center_y) * math.cos(theta) + center_y
				
				delta_x = (x0 - center_x)
				delta_y = (y0 - center_y)
				x = delta_x * angle_cos_array[idx] + delta_y * angle_sin_array[idx] + center_x
				y = -delta_x * angle_sin_array[idx] + delta_y * angle_cos_array[idx] + center_y
				
				x = math.floor(x)
				y = math.floor(y)

				if x >= (image.shape[1] - 1): x = image.shape[1] - 1
				elif x < 0: x = 0
				
				if y >= (image.shape[0] - 1): y = image.shape[0] - 1
				elif y < 0: y = 0

				return x, y

			shape_color_type = obj.find('shape').get('color')
			color_type = (1, 1, 255)
			if 'Red' == shape_color_type:
				color_type = (1, 1, 255)
			elif 'Green' == shape_color_type:
				color_type = (1, 255, 1)

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

			t_x00, t_y00 = rotate_point(x00, y00, center_x, center_y, angle_array[angle_idx], angle_idx)
			t_x11, t_y11 = rotate_point(x11, y11, center_x, center_y, angle_array[angle_idx], angle_idx)
			t_x01, t_y01 = rotate_point(x01, y01, center_x, center_y, angle_array[angle_idx], angle_idx) #(t_x11, t_y00)#
			t_x10, t_y10 = rotate_point(x10, y10, center_x, center_y, angle_array[angle_idx], angle_idx) #(t_x00, t_y11)#


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

			tree.write(xml_output_path + "/" + image_name + '_' + str(angle_array[angle_idx]) + "_rotate.xml", encoding='utf-8', xml_declaration=True)
			cv2.rectangle(image_mark, (t_xmin, t_ymin), (t_xmax, t_ymax), color_type, 5)
			cv2.circle(image_mark, (t_xmin, t_ymin), 5, (255, 1, 1), -1) #Blue
			cv2.circle(image_mark, (t_xmax, t_ymax), 5, (1, 255, 1), -1) #Green
			cv2.imwrite(image_output_path + "/" + image_name + '_' + str(angle_array[angle_idx])  + "_rotate_MARKED.jpg", image_mark)
			if 0:
				#cv2.rectangle(image_mark, (xmin, ymin), (xmax, ymax), (1, 1, 255), 5)
				#cv2.rectangle(image_mark, (t_xmin, t_ymin), (t_xmax, t_ymax), (1, 255, 1), 5)
				#cv2.circle(image_mark, (t_xmin, t_ymin), 5, (1, 255, 1), -1)
				#cv2.circle(image_mark, (t_xmax, t_ymax), 5, (1, 255, 1), -1)
				#cv2.circle(image_mark, (t_x00, t_y00), 5, (255, 1, 1), -1)
				#cv2.circle(image_mark, (t_x01, t_y01), 5, (1, 255, 1), -1)
				#cv2.circle(image_mark, (t_x10, t_y10), 5, (1, 255, 1), -1)
				#cv2.circle(image_mark, (t_x11, t_y11), 5, (1, 255, 1), -1)				
				cv2.namedWindow(str(angle_array[angle_idx]), 0)
				cv2.imshow(str(angle_array[angle_idx]), image_mark)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
	#break

#@jit
def augment_flip(flip, txt_input_path, image_input_path, xml_input_path, txt_output_path, image_output_path, xml_output_path):
			
	total_images = len(open(txt_input_path + "/train.txt",'rU').readlines())

	image_id = 0
	for image_name in open(txt_input_path + "/train.txt"):
		image_name = image_name.strip('\n')
		print(image_input_path + "/" + image_name + ".jpg")
		image = cv2.imread(image_input_path + "/" + image_name + ".jpg", -1)
		if image is None:
			print("image is NULL...")
			continue

		view_bar(image_id, total_images)
		image_id = image_id + 1
		
		image_flip = cv2.flip(image, flip)
		image_mark = image_flip.copy()
		cv2.imwrite(image_output_path + "/" + image_name + '_' + str(flip) + '_flip.jpg', image_flip)

		shutil.copyfile(xml_input_path + "/" + image_name + ".xml", xml_output_path + "/" + image_name + '_' + str(flip) + "_flip.xml")

		print(xml_output_path + "/" + image_name + '_' + str(flip) + "_flip.xml")
		tree = ET.parse(xml_output_path + "/" + image_name + '_' + str(flip) + "_flip.xml")
		root = tree.getroot()
		root.find('filename').text = image_name + '_' + str(flip) + '_flip.jpg'

		for obj in root.findall('object'):
			shape_color_type = obj.find('shape').get('color')
			color_type = (1, 1, 255)
			if 'Red' == shape_color_type:
				color_type = (1, 1, 255)
			elif 'Green' == shape_color_type:
				color_type = (1, 255, 1)

			xmin = int(int(obj.find('bndbox/xmin').text) / 1)
			ymin = int(int(obj.find('bndbox/ymin').text) / 1)
			xmax = int(int(obj.find('bndbox/xmax').text) / 1)
			ymax = int(int(obj.find('bndbox/ymax').text) / 1)
			
			if -1 == flip:#hv
				t_xmin = image.shape[1] - 1 - xmax
				t_ymin = image.shape[0] - 1 - ymax
				t_xmax = image.shape[1] - 1 - xmin
				t_ymax = image.shape[0] - 1 - ymin
			elif 0 == flip:#v
				t_xmin = xmin 
				t_ymin = image.shape[0] - 1 - ymax
				t_xmax = xmax
				t_ymax = image.shape[0] - 1 - ymin
			elif 1 == flip:#h
				t_xmin = image.shape[1] - 1 - xmax 
				t_ymin = ymin
				t_xmax = image.shape[1] - 1 - xmin
				t_ymax = ymax
				
			obj.find('bndbox/xmin').text = str(t_xmin)
			obj.find('bndbox/ymin').text = str(t_ymin)
			obj.find('bndbox/xmax').text = str(t_xmax)
			obj.find('bndbox/ymax').text = str(t_ymax)

			obj.findall('shape/points/x')[0].text = obj.find('bndbox/xmin').text
			obj.findall('shape/points/y')[0].text = obj.find('bndbox/ymin').text
			obj.findall('shape/points/x')[1].text = obj.find('bndbox/xmax').text
			obj.findall('shape/points/y')[1].text = obj.find('bndbox/ymax').text

			tree.write(xml_output_path + "/" + image_name + '_' + str(flip) + "_flip.xml", encoding='utf-8', xml_declaration=True)
			cv2.rectangle(image_mark, (t_xmin, t_ymin), (t_xmax, t_ymax), color_type, 5)
			cv2.circle(image_mark, (t_xmin, t_ymin), 5, (255, 1, 1), -1) #Blue
			cv2.circle(image_mark, (t_xmax, t_ymax), 5, (1, 255, 1), -1) #Green
			cv2.imwrite(image_output_path + "/" + image_name + '_' + str(flip)  + "_flip_MARKED.jpg", image_mark)
			if 0:
				#cv2.rectangle(image_mark, (xmin, ymin), (xmax, ymax), (1, 1, 255), 5)
				cv2.rectangle(image_mark, (t_xmin, t_ymin), (t_xmax, t_ymax), (1, 255, 1), 5)			
				cv2.namedWindow(str(flip), 0)
				cv2.imshow(str(flip), image_mark)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
	#break
#@jit
def augment_crop(crop, txt_input_path, image_input_path, xml_input_path, txt_output_path, image_output_path, xml_output_path):

	total_images = len(open(txt_input_path + "/train.txt",'rU').readlines())

	image_id = 0
	for image_name in open(txt_input_path + "/train.txt"):
		image_name = image_name.strip('\n')
		print(image_input_path + "/" + image_name + ".jpg")
		image = cv2.imread(image_input_path + "/" + image_name + ".jpg", -1)
		if image is None:
			print("image is NULL...")
			continue

		view_bar(image_id, total_images)
		image_id = image_id + 1

		crop_ratio = float(random.randint(75, 98)) / 100.0
		crop_size_x = int(image.shape[1] * crop_ratio)
		crop_size_y = int(image.shape[0] * crop_ratio)
		#print(crop_size_x, crop_size_y)

		random_x = random.randint(-int((image.shape[1] - crop_size_x)  / 10), int((image.shape[1] - crop_size_x) / 10))
		random_y = random.randint(-int((image.shape[0] - crop_size_y) / 10), int((image.shape[0] - crop_size_y) / 10))
		center_x = image.shape[1] / 2 + random_x
		center_y = image.shape[0] / 2 + random_y
		
		x0 = int(center_x - crop_size_x / 2)
		y0 = int(center_y - crop_size_y / 2)
		x1 = int(center_x + crop_size_x / 2)
		y1 = int(center_y + crop_size_y / 2) 
		image_crop = image[y0:y1, x0:x1]
						
		image_mark = image_crop.copy()
		cv2.imwrite(image_output_path + "/" + image_name + '_' + str(crop) + '_crop.jpg', image_crop)

		shutil.copyfile(xml_input_path + "/" + image_name + ".xml", xml_output_path + "/" + image_name + '_' + str(crop) + "_crop.xml")

		print(xml_output_path + "/" + image_name + '_' + str(crop) + "_crop.xml")
		tree = ET.parse(xml_output_path + "/" + image_name + '_' + str(crop) + "_crop.xml")
		root = tree.getroot()
		root.find('filename').text = image_name + '_' + str(crop) + '_crop.jpg'

		#image_size	
		#print(image_crop.shape[1], image_crop.shape[0], image_crop.shape[2])
		root.find("size/width").text = str(image_crop.shape[1])
		root.find("size/height").text = str(image_crop.shape[0])
		root.find("size/depth").text = str(image_crop.shape[2])

		for obj in root.findall('object'):
			shape_color_type = obj.find('shape').get('color')
			color_type = (1, 1, 255)
			if 'Red' == shape_color_type:
				color_type = (1, 1, 255)
			elif 'Green' == shape_color_type:
				color_type = (1, 255, 1)

			xmin_ = int(int(obj.find('bndbox/xmin').text) / 1)
			ymin_ = int(int(obj.find('bndbox/ymin').text) / 1)
			xmax_ = int(int(obj.find('bndbox/xmax').text) / 1)
			ymax_ = int(int(obj.find('bndbox/ymax').text) / 1)
			area_orin = (ymax_ - ymin_) * (xmax_ - xmin_)

			xmin = int(int(obj.find('bndbox/xmin').text) / 1) - x0
			ymin = int(int(obj.find('bndbox/ymin').text) / 1) - y0
			xmax = int(int(obj.find('bndbox/xmax').text) / 1) - x0
			ymax = int(int(obj.find('bndbox/ymax').text) / 1) - y0
			
			if xmin < 0: xmin = 0
			if ymin < 0: ymin = 0
			if xmax > image_crop.shape[1] - 1: xmax = image_crop.shape[1] - 1
			if ymax > image_crop.shape[0] - 1: ymax = image_crop.shape[0] - 1
			area = (ymax - ymin) * (xmax - xmin)

			if xmax < 0 or ymax < 0 or xmin > (image_crop.shape[1] - 1) or ymin > (image_crop.shape[0] - 1) or area < 0.5 * area_orin:
				xmin = 0
				ymin = 0
				xmax = 4
				ymax = 4

			#print(xmin, ymin, xmax, ymax)

			obj.find('bndbox/xmin').text = str(xmin)
			obj.find('bndbox/ymin').text = str(ymin)
			obj.find('bndbox/xmax').text = str(xmax)
			obj.find('bndbox/ymax').text = str(ymax)

			obj.findall('shape/points/x')[0].text = obj.find('bndbox/xmin').text
			obj.findall('shape/points/y')[0].text = obj.find('bndbox/ymin').text
			obj.findall('shape/points/x')[1].text = obj.find('bndbox/xmax').text
			obj.findall('shape/points/y')[1].text = obj.find('bndbox/ymax').text

			tree.write(xml_output_path + "/" + image_name + '_' + str(crop) + "_crop.xml", encoding='utf-8', xml_declaration=True)
			cv2.rectangle(image_mark, (xmin, ymin), (xmax, ymax), color_type, 5)
			cv2.circle(image_mark, (xmin, ymin), 5, (255, 1, 1), -1) #Blue
			cv2.circle(image_mark, (xmax, ymax), 5, (1, 255, 1), -1) #Green
			cv2.imwrite(image_output_path + "/" + image_name + '_' + str(crop)  + "_crop_MARKED.jpg", image_mark)
			if 0:
				cv2.rectangle(image_mark, (xmin, ymin), (xmax, ymax), (1, 255, 1), 5)			
				cv2.namedWindow(str(crop), 0)
				cv2.imshow(str(crop), image_mark)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
		#break
	
#@jit
def csvToxml():
	train_csv_file_name = "train_labels.csv"
	test_csv_file_name = "submit_example.csv"
	
	image_path = "data/VOCdevkit2007/VOC2007_origin/JPEGImages"
	xml_output_path = "data/VOCdevkit2007/VOC2007_origin/Annotations"
	trainval_dataset = "data/VOCdevkit2007/VOC2007_origin/ImageSets/Main"
	
	if_no_exist_path_and_make_path(image_path)
	if_no_exist_path_and_make_path(xml_output_path)
	if_no_exist_path_and_make_path(trainval_dataset)
		
	parse_csv(train_csv_file_name, image_path, xml_output_path, trainval_dataset, 1)
	#parse_csv(test_csv_file_name, image_path, xml_output_path, trainval_dataset, 0)

#@jit
def augment():
	txt_input_path = "data/VOCdevkit2007/VOC2007_origin/ImageSets/Main"
	image_input_path = "data/VOCdevkit2007/VOC2007_origin/JPEGImages"
	xml_input_path = "data/VOCdevkit2007/VOC2007_origin/Annotations"

	if args.is_rotate:
		angle_array = [args.rotation_angle]#[-180, -90, -6, -4, -2, 0, 2, 4, 6, 90]
		for angle_idx in range(0, len(angle_array)):
			print("angle: ", angle_array[angle_idx])
			
			txt_output_path = "data/VOCdevkit2007/" + "VOC2007_rotation_" + str(angle_array[angle_idx]) + "/ImageSets/Main"
			image_output_path = "data/VOCdevkit2007/" + "VOC2007_rotation_" + str(angle_array[angle_idx]) + "/JPEGImages"
			xml_output_path = "data/VOCdevkit2007/" + "VOC2007_rotation_" + str(angle_array[angle_idx]) + "/Annotations"
			
			if_no_exist_path_and_make_path(txt_output_path)
			if_no_exist_path_and_make_path(image_output_path)
			if_no_exist_path_and_make_path(xml_output_path)

			augment_rotation(angle_idx, angle_array, txt_input_path, image_input_path, xml_input_path, txt_output_path, image_output_path, xml_output_path)

	if args.is_flip:
		flip_array = [args.flip_type] #[-1, 0, 1]
		for flip in flip_array:
			print("flip: ", flip)

			txt_output_path = "data/VOCdevkit2007/" + "VOC2007_flip_" + str(flip) + "/ImageSets/Main"
			image_output_path = "data/VOCdevkit2007/" + "VOC2007_flip_" + str(flip) + "/JPEGImages"
			xml_output_path = "data/VOCdevkit2007/" + "VOC2007_flip_" + str(flip) + "/Annotations"
			
			if_no_exist_path_and_make_path(txt_output_path)
			if_no_exist_path_and_make_path(image_output_path)
			if_no_exist_path_and_make_path(xml_output_path)

			augment_flip(flip, txt_input_path, image_input_path, xml_input_path, txt_output_path, image_output_path, xml_output_path)

	if args.is_crop:
		crop_num = args.crop_num
		for crop in range(crop_num - 1, crop_num):
			print("crop: ", crop)
	
			txt_output_path = "data/VOCdevkit2007/" + "VOC2007_crop_" + str(crop) + "/ImageSets/Main"
			image_output_path = "data/VOCdevkit2007/" + "VOC2007_crop_" + str(crop) + "/JPEGImages"
			xml_output_path = "data/VOCdevkit2007/" + "VOC2007_crop_" + str(crop) + "/Annotations"
			
			if_no_exist_path_and_make_path(txt_output_path)
			if_no_exist_path_and_make_path(image_output_path)
			if_no_exist_path_and_make_path(xml_output_path)

			augment_crop(crop, txt_input_path, image_input_path, xml_input_path, txt_output_path, image_output_path, xml_output_path)

if __name__ == "__main__":
	#csvToxml()

	augment()

