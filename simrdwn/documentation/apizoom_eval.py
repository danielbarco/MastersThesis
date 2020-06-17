import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math

import numpy as np

# Apapted from https://github.com/Cartucho/mAP#create-the-ground-truth-files
##########
# This code was adapted so it accepts results from a SIMRDWN results dataframe.
# It is not exactly efficient to run. It may be more prudent to convert the 
# YOLO / VOC labels to COCO and then use the COCO API 
# http://cocodataset.org/#detection-eval
##########

"""
 Create a ".temp_files/" and "results/" directory
"""
TEMP_FILES_PATH = ".temp_files"
if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
    os.makedirs(TEMP_FILES_PATH)
results_files_path = "results"
if os.path.exists(results_files_path): # if it exist already
    # reset the results directory
    shutil.rmtree(results_files_path)

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

def get_ap_VOC(df_pred, df_gt, min_overlap = 0.5, threshhold = 0):

    TEMP_FILES_PATH = ".temp_files"
    if os.path.exists(TEMP_FILES_PATH): # if it exist already
        # reset the results directory
        shutil.rmtree(TEMP_FILES_PATH)
        os.makedirs(TEMP_FILES_PATH)
    else:
        os.makedirs(TEMP_FILES_PATH)

    """
    Calculate the AP for each class
    """
    df_pred = df_pred[df_pred['Prob'] >= threshhold]
    print('minimal overlap: ' + str(min_overlap))
    print('threshhold: ' + str(threshhold))
    bounding_boxes = []
    already_seen_classes = []
    gt_counter_per_class = {}
    counter_images_per_class = {}   
    det_counter_per_class = {}
    df_gt['overlap'], df_gt['used'], df_gt['pred_index'] = None, None, None
    df_pred['gt_index'] = None
    df_pred.sort_values(by='Prob', ascending=False, inplace = True)
    for index, row in df_pred.iterrows(): 
        bbox = str(row['x']) + " " + str(row['y']) + " " + str(row['width']) + " " + str(row['height'])
        bounding_boxes.append({"confidence": str(row['Prob']), "file_id": row['Image_Root'], "bbox": bbox, 'pred_index' : index, 'gt_index' : index})
        """
         Count total of detection-results
        """
        class_name = str(row['class'])
        # # check if class is in the ignore list, if yes skip
        # if class_name in args.ignore:
        #     continue
        # count that object
        if class_name in det_counter_per_class:
            det_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            det_counter_per_class[class_name] = 1
    file_name_list = df_gt['filename'].unique()
    for file_name in file_name_list:
        bounding_boxes2 = []
        gt_file = TEMP_FILES_PATH + file_name + "_ground_truth.json" 
        for index, row in df_gt.loc[df_gt['filename'] == file_name].iterrows():  
            class_name = str(row['class'])
            bbox2 = str(row['x']) + " " + str(row['y']) + " " + str(row['width']) + " " + str(row['height'])
            bounding_boxes2.append({"class_name": class_name, "bbox": bbox2, "used":False, "file_id": file_name, 'gt_index' : index, 'pred_index' : index, 'overlap' : 0})
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1

            if class_name not in already_seen_classes:
                if class_name in counter_images_per_class:
                    counter_images_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_name] = 1
                already_seen_classes.append(class_name)
            with open(gt_file, 'w') as outfile:
                json.dump(bounding_boxes2, outfile)


    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # open file to store the results
    ##with open(results_files_path + "/results_new.txt", 'w') as results_file:
    # with open("/results_new.txt", 'w') as results_file:
    #     results_file.write("# AP and precision/recall per class\n")
    count_true_positives = {}
    mAP = 0
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        """
        Load detection-results of that class
        """
        ##dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data =   bounding_boxes ##json.load(open(dr_file))

        """
        Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        print('nd: ', nd)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["file_id"]
            gt_file = TEMP_FILES_PATH + file_id + "_ground_truth.json"
            ground_truth_data =  json.load(open(gt_file))
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = [ float(x) for x in detection["bbox"].split() ]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = [ float(x) for x in obj["bbox"].split() ]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
            if ovmax >= min_overlap:
                if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            gt_match["overlap"] = ovmax
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                            # if show_animation:
                            #     status = "MATCH!"
                            df_gt.loc[[gt_match["gt_index"]], 'overlap'] = gt_match["overlap"]
                            df_gt.loc[[gt_match["gt_index"]], 'used'] = 1
                            df_gt.loc[[gt_match["gt_index"]], 'pred_index'] = detection['pred_index']
                            df_pred.loc[[detection['pred_index']], 'gt_index'] = gt_match["gt_index"]
                            
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1

            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

        #print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        print('fp -1: ',fp[-1])
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        print('tp -1: ',tp[-1])
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        #print('recall: ' + str(len(rec)))
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        print('ap: ', ap)
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
        """
        Write to results.txt
        """
        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]
        #results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        #if not args.quiet:
        print(text)
        ap_dictionary[class_name] = ap

        n_images = counter_images_per_class[class_name]
        lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
        lamr_dictionary[class_name] = lamr

        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP*100)
        # results_file.write(text + "\n")
        print(text)

        #print(det_counter_per_class)
        dr_classes = list(det_counter_per_class.keys())

        TP = count_true_positives[class_name]
        n_det = det_counter_per_class[class_name]
        FP = n_det - count_true_positives[class_name]
        FN = gt_counter_per_class[class_name] - count_true_positives[class_name]
        print("\n# Number of ground-truth objects per class")
        for class_name in sorted(gt_counter_per_class):
                print(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")
        print("TP: " + str(TP) + ', FP: ' + str(FP) + '\n' + 'FN: ' + str(FN))
        print( "\n Precision: " + str(TP/(TP+FP)) + "\n Recall :" + str(TP/(TP+FN)) +  "\n F1-score: " + str(2 * TP/ (2 * TP + FP + FN)) + "\n")
        print(len(mprec))   
        print('-------------------------------------------------------')
    
    # remove the temp_files directory
    shutil.rmtree(TEMP_FILES_PATH)

    return df_gt, df_pred, TP, FP, FN, ap, mrec, mprec, tp, fp

###################################################

def get_ap_COCO_avg(df_pred, df_gt, threshhold = 0):
    coco_AP_VOC_results = []
    coco_AP_results = []
    coco_AP = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    for IoU in coco_AP:
        df_gt0, df_pred0, TP, FP, FN, ap, mrec, mprec, rec, prec = get_ap_VOC(df_pred, df_gt, IoU, threshhold)
        coco_AP_VOC_results.append(ap)
        coco_AP_results.append(TP/(TP+FP))
    ap_coco_avg = sum(coco_AP_results) / float(len(coco_AP_results))
    print(ap_coco_avg)
    return ap_coco_avg   

###################################################
