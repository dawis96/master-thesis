import multiprocessing as mp
import cv2
import numpy as np

from detectron2.config import get_cfg

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set model
    cfg.MODEL.WEIGHTS = args.weights
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def compute_polygon_area(points):
    s = 0
    point_num = len(points)
    if(point_num < 3): return 0.0
    for i in range(point_num): 
        s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
    return abs(s/2.0)

def save_result_to_txt(txt_save_path,prediction,polygons, img, img_save_path):

    file = open(txt_save_path,'w')

    classes = prediction['instances'].pred_classes
    cropped_texts = []
    idx = 0
    for i in range(len(classes)):
        if classes[i]==0:
            if len(polygons[i]) != 0:
                points = []
                for j in range(0,len(polygons[i][0]),2):
                    points.append([polygons[i][0][j],polygons[i][0][j+1]])
                points = np.array(points)
                area = compute_polygon_area(points)
                rect = cv2.minAreaRect(points)
                box = cv2.boxPoints(rect)

                box = box.clip(min=0)
                if area > 175:
                    file.writelines(str(int(box[0][0]))+','+str(int(box[0][1]))+','+str(int(box[1][0]))+','+str(int(box[1][1]))+','
                              +str(int(box[2][0]))+','+str(int(box[2][1]))+','+str(int(box[3][0]))+','+str(int(box[3][1])))
                    file.write('\r\n')

                    box = np.int0(box)
                    crop_box = img[min(box[:, 1]):max(box[:, 1]), min(box[:, 0]):max(box[:, 0])]
                    # cv2.drawContours(img, [np.int0(box)], 0, (0, 0, 255), 1)
                    img_path = img_save_path[:-4]+"_"+str(idx)+".jpg"
                    cv2.imwrite(img_path, crop_box)  
                    cropped_texts.append(crop_box)
                    idx += 1

    file.close()
    return cropped_texts