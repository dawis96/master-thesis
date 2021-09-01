import argparse
import os
import torch
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
import glob
from tqdm import tqdm
import json

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


from recognition_utils import *
# constants
WINDOW_NAME = "COCO detections"


def main():
    args = {
        'exp': "./experiments/seg_detector/totaltext_resnet50_deform_thre.yaml",
        'resume': "./weights/ic15_resnet50",
        'image_path': "/home/damiangrzywna/magisterka/icdar2015_test_images/*jpg",
        "result_dir": "/home/damiangrzywna/magisterka",
        'image_short_side': 736,
        'box_thresh': 0.5,
        'visualize': True,
        'resize': False,
        'polygon': False,
        'eager_show': False,
        'recognition_weights':"./weights/demo.pth.tar"
    }
        
    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    demo = Demo(experiment, experiment_args, cmd=args)
    demo.device = torch.device('cuda')
    detection_model = demo.init_model()
    demo.resume(detection_model, args['resume'])
        # all_matircs = {}
    detection_model.eval()

    dataset_info = DataInfo("ALLCASES_SYMBOLS")
    recognition_model = ModelBuilder(arch="ResNet_ASTER", 
                                    rec_num_classes=dataset_info.rec_num_classes,
                       sDim=512, attDim=512, max_len_labels=100,
                       eos=dataset_info.char2id[dataset_info.EOS], STN_ON=True)


    checkpoint = load_checkpoint(args['recognition_weights'])

    recognition_model.load_state_dict(checkpoint['state_dict'])
    device = torch.device("cuda")
    recognition_model = recognition_model.to(device)
    recognition_model = nn.DataParallel(recognition_model)
    recognition_model.eval()

    prediction_times = []
    recognition_times = []
    total_time = []
    for img_path in tqdm(glob.glob(args['image_path'])):
        start_time1 = time.time()
        boxes = demo.inference(detection_model, img_path)
        prediction_time = time.time() - start_time1
        prediction_times.append(prediction_time)
        img = cv2.imread(img_path)
        cropped_texts = []
        for i, box in enumerate(boxes):
            crop_box = img[min(box[:, 1]):max(box[:, 1]), min(box[:, 0]):max(box[:, 0])]
            # result_file_name = 'detection_res_' + img_path.split('/')[-1].split('.')[0]+"_" + str(i) + '.jpg'
            # result_file_path = os.path.join(args['result_dir'], result_file_name)
            # img_path = img_save_path[:-4]+"_"+str(idx)+".jpg"
            # cv2.imwrite(result_file_path, crop_box) 
            cropped_texts.append(crop_box)
    
        

        # recognition_file_name = result_file_name = 'recognition_res_' + img_path.split('/')[-1].split('.')[0]+'.txt'
        # recognition_file_path = os.path.join(args['result_dir'], recognition_file_name)
        # file = open(recognition_file_path,'w')
        start_time2 = time.time()
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
        #     file.writelines('{0}'.format(pred_str[0]))
        #     file.write('\r\n')
        # file.close()
        recognition_time = time.time() - start_time2
        recognition_times.append(recognition_time)
        total_time.append(time.time() - start_time1)
    
    times = {
        "prediction": prediction_times,
        "recognition": recognition_times,
        "total": total_time
    }
    with open(os.path.join(args['result_dir'], "infer_time_db_ic15.json"), 'w') as outfile:
        json.dump(times, outfile)

class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output):
        boxes_list = []
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'detection_res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        
                        box = boxes[i,:,:].reshape(-1).tolist()
                        boxes_list.append(boxes[i,:,:])
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
        return boxes_list

    def inference(self, model, image_path, visualize=False):
        self.init_torch_tensor()
        # model = self.init_model()
        # self.resume(model, self.model_path)
        # all_matircs = {}
        # model.eval()
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            boxes = self.format_output(batch, output)
            return boxes
            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                print(image_path)
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)

if __name__ == '__main__':
    main()