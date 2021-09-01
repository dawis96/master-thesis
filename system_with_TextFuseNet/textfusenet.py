# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import json
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

#ASTER imports

import sys
sys.path.append('./')

import argparse
import os
import time

import torch
from torch import nn
from lib.models.model_builder import ModelBuilder
from lib.utils.serialization import load_checkpoint
from lib.evaluation_metrics.metrics import get_str_list


from detection_utils import *
from recognition_utils import *
# constants
WINDOW_NAME = "COCO detections"

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="/home/damiangrzywna/magisterka/textfusenet_pipeline/icdar2015_101_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--weights",
        default="/home/damiangrzywna/magisterka/textfusenet_pipeline/weights/model_ic15_r101.pth",
        metavar="pth",
        help="the model used to inference",
    )

    parser.add_argument(
        "--recognition_weights",
        default="/home/damiangrzywna/magisterka/textfusenet_pipeline/weights/demo.pth.tar",
    )

    parser.add_argument(
        "--input",
        default="/home/damiangrzywna/magisterka/totaltext_test_images/part_1/*jpg",
        nargs="+",
        help="the folder of icdar2015 test images"
    )

    parser.add_argument(
        "--output",
        default="/home/damiangrzywna/magisterka/totaltext_test_images/textfusenet_ic15_weights/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser




if __name__ == "__main__":
   
    args = get_parser().parse_args()
    
    cfg = setup_cfg(args)
    detection_model = VisualizationDemo(cfg)
    
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
   
   
    dataset_info = DataInfo("ALLCASES_SYMBOLS")



    recognition_model = ModelBuilder(arch="ResNet_ASTER", 
                                    rec_num_classes=dataset_info.rec_num_classes,
                       sDim=512, attDim=512, max_len_labels=100,
                       eos=dataset_info.char2id[dataset_info.EOS], STN_ON=True)


    checkpoint = load_checkpoint(args.recognition_weights)

    recognition_model.load_state_dict(checkpoint['state_dict'])
    device = torch.device("cuda")
    recognition_model = recognition_model.to(device)
    recognition_model = nn.DataParallel(recognition_model)
    recognition_model.eval()

    test_images_path = args.input
    output_path = args.output

    start_time_all = time.time()
    img_count = 0
    prediction_times = []
    recognition_times = []
    for i in tqdm(glob.glob(test_images_path)):

        img_name = os.path.basename(i)

        start_time = time.time()
        img = cv2.imread(i)
        # print(img.shape)
        if img.shape[0] > 1900:
            # print("reshaping")
            scale_percent = 50

            #calculate the 50 percent of original dimensions
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)

            # dsize
            dsize = (width, height)

            # resize image
            img = cv2.resize(img, dsize)


        
        prediction, vis_output, polygons = detection_model.run_on_image(img)

        txt_save_path = output_path + 'detection_res_' + img_name.split('.')[0] + '.txt'
        rec_txt_save_path = output_path + 'recognition_res_' + img_name.split('.')[0] + '.txt'
        img_save_path = output_path + 'detection_res_' + img_name.split('.')[0] + '.jpg'
        cropped_texts = save_result_to_txt(txt_save_path, prediction, polygons, img, img_save_path) 

        prediction_time = time.time() - start_time

        # vis_output.save(img_save_path)
        img_count += 1
        prediction_times.append(prediction_time)
        # RECOGNITION
        file = open(rec_txt_save_path,'w')
        start_time = time.time()
        for text_img in cropped_texts:
            text_img = image_process(text_img)
            with torch.no_grad():
                text_img = text_img.to(device)
            input_dict = {}
            input_dict['images'] = text_img.unsqueeze(0)
            # TODO: testing should be more clean.
            # to be compatible with the lmdb-based testing, need to construct some meaningless variables.
            rec_targets = torch.IntTensor(1, 100).fill_(1)
            rec_targets[:,100-1] = dataset_info.char2id[dataset_info.EOS]
            input_dict['rec_targets'] = rec_targets
            input_dict['rec_lengths'] = [100]
            output_dict = recognition_model(input_dict)
            pred_rec = output_dict['output']['pred_rec']
            pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
            file.writelines('{0}'.format(pred_str[0]))
            file.write('\r\n')
        file.close()
        recognition_time = time.time() - start_time
        recognition_times.append(recognition_time)
    
    times = {
        "prediction": prediction_times,
        "recognition": recognition_times
    }
    with open( output_path + "infer_time.json", 'w') as outfile:
        json.dump(times, outfile)