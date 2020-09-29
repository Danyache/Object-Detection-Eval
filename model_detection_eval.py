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
            try:
                recalls.append(mc['recall'][-1])
            except:
                recalls.append(0)
            maps.append(mc['AP'])

        mAR += np.array(recalls).mean()
        mAP_over += np.array(maps).mean()

    mAR /= len(full_iouts)
    mAP_over /= len(full_iouts)

    columns = ['classes']
    class_names = []
    
    for iou_t in iou_ts:
        for col_name in ['AP', 'ACC', 'TP', 'FP', 'total_P', 'Pr', 'Re']:
            columns.append(f'{col_name}@{int(iou_t*100)}')
    
    row_dict = {}
    
    loop_idx = 0
    
    for iou_t in iou_ts:
        
        loop_idx += 1
        
        metricsPerClass = evaluator.GetPascalVOCMetrics(
            all_bboxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=iou_t,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation)

        average_precisions = []
        true_positives = []
        total_positives = []
        false_positives = []
        
        for mc in metricsPerClass:

            # Get metric values per each class
            c = mc['class']
            
            if loop_idx == 1:
                class_names.append(c)
                row_dict[c] = []
            
            # row_dict[c].append(iou_t)
            
            precision = mc['precision']
            recall = mc['recall']
            average_precision = mc['AP']
#             class_dict['AP'] = average_precision
            row_dict[c].append(average_precision)
            
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
            
            row_dict[c].append(TP/(FP+total_p))
            row_dict[c].append(TP)
            row_dict[c].append(FP)
            row_dict[c].append(total_p)
            
            if TP+FP>0:
                row_dict[c].append(TP/(TP+FP))
            else:
                row_dict[c].append(0)
                
            row_dict[c].append(TP/total_p)
        
        if loop_idx == 1:
            row_dict['mAP'] = []
            row_dict['Mean ACC'] = []
        
        row_dict['mAP'].append(np.array(average_precisions).mean())
        row_dict['Mean ACC'].append(np.array(true_positives).sum() / (np.array(total_positives).sum() + np.array(false_positives).sum()))
        
    
    row_list = []
    for c in class_names:
        c_arr = row_dict[c]
        c_arr.insert(0, c)
        row_list.append(c_arr)
        
    row_list.append([])
    
    mAP_list = row_dict['mAP']
    mACC_list = row_dict['Mean ACC']
    
    mAP_list.insert(0, 'mAP')
    mACC_list.insert(0, 'Mean ACC')
    
    iou_list = [iou_t for iou_t in iou_ts]
    iou_list.insert(0, 'IOU_t')
    
    row_list.append(iou_list)
    row_list.append(mAP_list)
    row_list.append(mACC_list)
    
    row_list.append([])
    row_list.append(['mAP over iou=[.5-.95]', mAP_over])
    row_list.append(['mAR', mAR])
        
    res_df = pd.DataFrame(row_list, columns=columns)
    
    res_df.to_csv(output_file, index=False)

if __name__=='__main__':
	"""
	Пример ввода:
	python model_detection_eval.py -gt_path val_small_40_classes.json -pred_path coco_instances_results.json -output_path output3.csv -t "0.5 0.6 0.9"
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
			default='output.csv',
			metavar='',
			help='Path to the file with the results')

	args = parser.parse_args()

	main(args)