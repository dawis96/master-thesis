import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def count_detection_metrics(df):
    TP = len(df[df.detection_status == "TP"])
    FP = len(df[df.detection_status == "FP"])
    FN = len(df[df.detection_status == "FN"])
    
    precision = round( (TP / (TP + FP)) * 100, 2) if (TP + FP) > 0 else 0
    recall = round( (TP / (TP + FN)) * 100, 2) if (TP + FN) > 0 else 0
    f1 = round( ((2 * precision * recall) / (precision + recall)), 2) if (precision + recall) > 0 else 0
    return precision, recall, f1

def detection_metrics_to_df(df_all, models_list, weights_list):
    
    detection_metrics = []
    for model in models_list:
        for weights in weights_list:
            df = df_all[(df_all.model == model) & (df_all.weights == weights)]
            precision, recall, f1 = count_detection_metrics(df)
            metrics = {
                    'system': f"System z modelem {model} wytrenowany na zbiorze {weights}",
    #                 'model': model,
    #                 'wagi': weight,
                    'precyzja': precision,
                    'pełność': recall,
                    'f1': f1
                }
            detection_metrics.append(metrics)
    detection_metrics = pd.DataFrame(detection_metrics)
    return detection_metrics

def plot_detection_metrics(detection_metrics, filename=None, title=""):
    fig, ax = plt.subplots( figsize=(10, 5))

    # detection_metrics.plot(kind='bar', x='label', y=['precision', 'recall', 'f1'], ax=ax)
    available_columns = ['system', 'precyzja', 'pełność', 'f1']
    
    detection_metrics[[col for col in detection_metrics.columns if col in available_columns]].set_index('system').T.plot(kind='bar', ax=ax, rot=0)

    ax.set_xlabel("Miara jakości")
    ax.set_ylabel("Wartość")
    ax.set_title(title)
    ax.legend(loc=4)

    ax.set_yticks(np.arange(0, 110, 10))
    # ax.set_xticklabels(ax.get_xticks(), rotation = 90)
    ax.grid(True)
    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename)
        
        
def plot_detection_metrics_in_one(detection_metrics, filename=None, category='', figsize=(15, 10)):
    df = []
    for key, value in detection_metrics.items():
        value[category] = key
        df.append(value)
    df = pd.concat(df)

    available_metrics =['precyzja', 'pełność', 'f1']
    metrics = [col for col in df.columns if col in available_metrics]
    
    fig, axs = plt.subplots(len(metrics), figsize=figsize)
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    
    for i, metric in enumerate(metrics):

        sns.pointplot(x=df[category], y=df[metric], hue=df.system, ax=axs[i],
                       markers="*", linestyles='--', errwidth=2, s=250, dodge=True, legend=0)
    # #         axs[ix, iy].axis('off')
        axs[i].set_title(f"{metric} detekcji tekstu dla każdego z systemów")
        axs[i].get_legend().remove()
#         axs[i].grid(True)

    handles, labels = axs[i].get_legend_handles_labels()
    fig.legend(handles, labels)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)

    # fig.legend( axs[i].lines, axs[i].labels, loc = (0.5, 0), ncol=5 )?
    plt.show()
    if filename is not None:
        fig.savefig(filename)

def plot_detections(df, models_list, weights_list, filename=None, linewidth=2):
    image_name, dataset = df[['image_name', 'dataset']].sample(n=1).values[0]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    paths = {
        "ic03": "../../icdar2003/SceneTrialTest",
        "ic15": "../../icdar2015/ch4_test_images",
        "tt": "../../total-text/Images/Test"
    }

    for ix, model in enumerate(models_list):
        for iy, weight in enumerate(weights_list):
            
            model_detections = df[(df.image_name == image_name) &
                                      (df.model == model) &
                                      (df.weights == weight)]
            precision, recall, f1 = count_detection_metrics(model_detections)
            
            img = cv2.imread(os.path.join(paths[dataset], image_name))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for i, row in model_detections.iterrows():
                if row['x1'] != '-':
    #                 print((row['x1'], row['y1']), (row['x2'], row['y2']))
                    color = (255, 0, 0)
                    cv2.rectangle(img, (int(row['x1']), int(row['y1'])), (int(row['x2']), int(row['y2'])) ,color, linewidth)

                if row['pred_x1'] != '-':  
                    color = (0, 255, 0)
                    cv2.rectangle(img, (int(row['pred_x1']), int(row['pred_y1'])), (int(row['pred_x2']), int(row['pred_y2'])) ,color, linewidth)

            axs[ix, iy].axis('off')
            axs[ix, iy].set_title(f"System z modelem {model} wytrenowanym na zbiorze {weight}\nJakość detekcji: precyzja={precision}, pełność={recall}, f1={f1}")

            axs[ix, iy].imshow(img);
            axs[ix, iy].grid(False)
    plt.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename)
        
#### RECOGNITION
        

def count_recognition_metrics(df, only_tp_detections=False):
    
    if only_tp_detections:
        df = df[df.detection_status == "TP"]
    else:
        df = df[df.detection_status.isin(["TP", 'FN'])]
    correctly_clasified = len(df[df.recogniton_status == 'Correct'])
    missclassified = len(df[df.recogniton_status == 'Wrong'])
    
    acc = round(correctly_clasified / (len(df)) * 100, 2) if len(df) > 0 else np.nan
    return acc

def recognition_metrics_to_df(df_all, models_list, weights_list):
    
    recognition_metrics = []
    for model in models_list:
        for weights in weights_list:
            df = df_all[(df_all.model == model) & (df_all.weights == weights)]
            acc_on_tp = count_recognition_metrics(df, only_tp_detections=True)
            acc_overall = count_recognition_metrics(df, only_tp_detections=False)
            metrics = {
                    'system': f"System z modelem {model} z wagami {weights}",
                    'dokładność na poprawnych detekcjach': acc_on_tp,
                    'dokładność ogólnie': acc_overall,
                }
            recognition_metrics.append(metrics)
    recognition_metrics = pd.DataFrame(recognition_metrics)
    return recognition_metrics

def plot_recognition_metrics(recognition_metrics, filename=None, title=""):
    fig, ax = plt.subplots( figsize=(10, 5))

    # detection_metrics.plot(kind='bar', x='label', y=['precision', 'recall', 'f1'], ax=ax)
    recognition_metrics.set_index('system').T.plot(kind='bar', ax=ax, rot=0)

    ax.set_xlabel("Miara jakości")
    ax.set_ylabel("Wartość")
    ax.set_title(title)
    ax.legend(loc=4)

    # ax.set_yticks(np.arange(0, 100, 10))
    # ax.set_xticklabels(ax.get_xticks(), rotation = 90)
    ax.grid(True)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
        
## IOU histogram
def plot_iou_histogram(df, models_list, weights_list, filename=None, figsize=(15, 10)):

    fig, axs = plt.subplots(2, 2, figsize)

    for ix, model in enumerate(models_list):
        for iy, weight in enumerate(weights_list):
            ious = df[(df.detection_status == "TP") &
                                      (df.model == model) &
                                      (df.weights == weight)]['iou']
            
            ious.plot.hist(ax=axs[ix, iy])
    #         axs[ix, iy].axis('off')
            axs[ix, iy].set_title(f"System z modelem {model} wytrenowanym na zbiorze {weight}")
            axs[ix, iy].set_xlabel("IoU")
            axs[ix, iy].set_ylabel("Częstość wystąpień")


    #         axs[ix, iy].imshow(img);
    #         axs[ix, iy].grid(False)
    plt.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename)
        
def plot_recognition_metrics_in_one(recognition_metrics, filename=None, figsize=(15, 10), category=''):
    df = []
    for key, value in recognition_metrics.items():
        value[category] = key
        df.append(value)
    df = pd.concat(df)

    # sns.scatterplot(x=df.category, y=df.precyzja, hue=df.system)

    

    metrics = list(df.columns)[1:-1]
    fig, axs = plt.subplots(len(metrics), figsize=figsize)
    for i, metric in enumerate(metrics):

        sns.pointplot(x=df[category], y=df[metric], hue=df.system, ax=axs[i],
                       markers="*", linestyles='--', errwidth=2, s=250, dodge=True, legend=0)
    # #         axs[ix, iy].axis('off')
        axs[i].set_title(f"{metric} rozpoznawania tekstu dla każdego z systemów")
        axs[i].get_legend().remove()
#         axs[i].grid(True)

    handles, labels = axs[i].get_legend_handles_labels()
    fig.legend(handles, labels)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)

    # fig.legend( axs[i].lines, axs[i].labels, loc = (0.5, 0), ncol=5 )?
    plt.show()
    if filename is not None:
        fig.savefig(filename)
        


def plot_recognitions(df_org, models_list, weights_list, filename=None):
    
 
    #df = df_org[(df_org.detection_status == "TP") & (df_org.recogniton_status=="Wrong")]
    #if len(df) == 0:
    df = df_org[(df_org.detection_status == "TP")]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    paths = {
        "ic03": "../../icdar2003/SceneTrialTest",
        "ic15": "../../icdar2015/ch4_test_images",
        "tt": "../../total-text/Images/Test"
    }
    
    while True:
        img_count = 0
        image_name, dataset, text = df[['image_name', 'dataset', 'text']].sample(n=1).values[0]

        for ix, model in enumerate(models_list):
            for iy, weight in enumerate(weights_list):

                model_detections = df[(df.image_name == image_name) &
                                          (df.model == model) &
                                          (df.weights == weight) &
                                         (df.text == text)]
                sample = model_detections.squeeze()

                if sample.empty or len(model_detections) != 1 or sample['pred_x1'] == "-":
                    continue

                img = cv2.imread(os.path.join(paths[dataset], image_name))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = img[int(sample['pred_y1']):int(sample['pred_y2']), int(sample['pred_x1']):int(sample['pred_x2'])]


                axs[ix, iy].set_title(f"System z modelem {model} wytrenowanym na zbiorze {weight}\nTekst: {sample['text']} Predykcja: {sample['pred_text']} Stauts: {sample['recogniton_status']}")

                axs[ix, iy].imshow(img);
                axs[ix, iy].grid(False)
                img_count += 1
        if img_count >= 4:
            break
    plt.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename)