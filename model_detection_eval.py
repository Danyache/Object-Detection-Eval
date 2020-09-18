import os
import pandas as pd
import numpy as np
import json
import argparse

import cv2
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *


def main(args):
	gt_path = args.gt_path
	pred_path = args.pred_path
	iou_ts = args.iou_ts
	output_file = args.output_path

	

	if type(iou_ts) == float:
		iou_ts = [iou_ts]
	else:
		iou_ts = [float(i) for i in iou_ts.split()]

	with open(gt_path) as json_file:
		gt_data = json.load(json_file)

	with open(pred_path) as json_file:
		res_data = json.load(json_file)

	all_bboxes = BoundingBoxes()

	img2size = {}

	for image in gt_data['images']:
		img2size[image['id']] = (image['height'], image['width'])

	id2ctg = {}

	for category in gt_data['categories']:
		id2ctg[category['id']] = category['name'] 

	for gt_idx, gt_obj in enumerate(gt_data['annotations']):
	
		img_id = gt_obj['image_id']
		gt_bbox = gt_obj['bbox']
		category_id = gt_obj['category_id']
		
		x, y, w, h = gt_bbox
		
		gt_boundingBox = BoundingBox(imageName=str(img_id), classId=id2ctg[category_id], x=x, y=y, 
								   w=w, h=h, typeCoordinates=CoordinatesType.Absolute,
								   bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=img2size[img_id])
		
		all_bboxes.addBoundingBox(gt_boundingBox)

	for detected_idx, detected_obj in enumerate(res_data):
		
		img_id = detected_obj['image_id']
		detected_bbox = detected_obj['bbox']
		category_id = detected_obj['category_id']
		score = detected_obj['score']
		
		x, y, w, h = detected_bbox
		
		gt_boundingBox = BoundingBox(imageName=str(img_id), classId=id2ctg[category_id],classConfidence=score, x=x, y=y, 
								   w=w, h=h, typeCoordinates=CoordinatesType.Absolute,
								   bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=img2size[img_id])
		
		all_bboxes.addBoundingBox(gt_boundingBox)

	evaluator = Evaluator()

	mAR = 0

	full_iouts = np.arange(0.5, 1., 0.05)

	mAR = 0
	mAP_over = 0

	for iou in full_iouts:
		recalls = []
		maps = []
		
		metricsPerClass = evaluator.GetPascalVOCMetrics(
		all_bboxes,  # Object containing all bounding boxes (ground truths and detections)
		IOUThreshold=iou,  # IOU threshold
		method=MethodAveragePrecision.EveryPointInterpolation)
		
		for mc in metricsPerClass:
			recalls.append(mc['recall'][-1])
			maps.append(mc['AP'])
		
		mAR += np.array(recalls).mean()
		mAP_over += np.array(maps).mean()

	mAR /= len(full_iouts)
	mAP_over /= len(full_iouts)

	with open(output_file, 'a') as f:
		for iou_t in iou_ts:
			metricsPerClass = evaluator.GetPascalVOCMetrics(
				all_bboxes,  # Object containing all bounding boxes (ground truths and detections)
				IOUThreshold=iou_t,  # IOU threshold
				method=MethodAveragePrecision.EveryPointInterpolation)
			
			f.write(f'IOU: {iou_t}\n')
			
			average_precisions = []
			true_positives = []
			total_positives = []
			false_positives = []
			
			for mc in metricsPerClass:
				# Get metric values per each class
				c = mc['class']
				precision = mc['precision']
				recall = mc['recall']
				average_precision = mc['AP']
				ipre = mc['interpolated precision']
				irec = mc['interpolated recall']
				
				# Print AP per class
				total_p = mc['total positives']
				TP = mc['total TP']
				FP = mc['total FP']
				
				average_precisions.append(average_precision)
				true_positives.append(TP)
				total_positives.append(total_p)
				false_positives.append(FP)
				
				f.write(f'{c}: AP={average_precision}, ACC={TP/(FP+total_p)}\n')
			
			f.write(f'mAP: {np.array(average_precisions).mean()} \n')
			f.write(f'mAP over iou=[.5-.95]: {mAP_over} \n')
			f.write(f'Accuracy: {np.array(true_positives).sum() / (np.array(total_positives).sum() + np.array(false_positives).sum())} \n')
			f.write(f'mAR: {mAR} \n\n')



if __name__=='__main__':
	"""
	Пример ввода:
	python model_detection_eval.py -gt_path val_small_40_classes.json -pred_path coco_instances_results.json -output_path output3.txt -t "0.5 0.6 0.9"
	"""

	parser = argparse.ArgumentParser(prog='Object Detection Metrics', description="some metrics")
	parser.add_argument(
			'-t',
			'--threshold',
			dest='iou_ts',
			default=0.5,
			metavar='',
			help='IOU threshold. Default 0.5. If you want to use several values of IOU write them in a string space-separated (ex.: "0.5 0.6")')

	parser.add_argument(
			'-gt_path',
			dest='gt_path',
			type=str,
			metavar='',
			required=True,
			help='Path to the file with Ground Truth bboxes')

	parser.add_argument(
			'-pred_path',
			dest='pred_path',
			type=str,
			metavar='',
			required=True,
			help='Path to the file with predicted bboxes')

	parser.add_argument(
		'-output_path',
			dest='output_path',
			type=str,
			default='output.txt',
			metavar='',
			help='Path to the file with the results')

	args = parser.parse_args()

	main(args)


