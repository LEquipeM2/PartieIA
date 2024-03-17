import os
import SoftSeg as segmentation
from pyannote.core import Timeline
from tqdm.notebook import  tqdm_notebook as tqdm
from pyannote.database.util import load_rttm, load_uem
from pyannote.core import Segment
import numpy as np
import torch
from torch import avg_pool1d
import random
import shutil

def find_low_confiance_frames(sample,segmentation, threshold, window_size, annotated_ratio, mode, file,keep_method):
    """
    Identify frames with confiance lower than the specified threshold using a sliding window.

    Parameters:
    - segmentation: segmentationFeature
        segmentationFeature containing the probabilities of each speaker.
    - threshold: float
        confiance threshold, below which frames are considered low confiance.
    - window_size: float
        Size of the sliding window in seconds.
    - annotated_ratio: floatl
        Ratio of the lowest confiance frames to return.

    Returns:
    - list of segments
        List of segments with low confiance.
    """

    def sliding_window(elements, window_size,step):    
        if len(elements) <= window_size:
            return elements
        for i in range(0,len(elements)- window_size + 1,step):
            yield elements[i:i+window_size]
        #add the last window
        if len(elements) % window_size != 0:
            yield elements[-window_size:]
    
    segments = []
    window_size = int(window_size/segmentation.sliding_window.step)
    windows = sliding_window(segmentation.data,window_size,window_size)
    
    for i,window in enumerate(windows):
        window = np.nan_to_num(window)
        maxi_prob = np.max(window)
        second_maxi_prob = np.sort(window)[-2]
        confiance = maxi_prob - second_maxi_prob
        confiance_moyenne = np.mean(confiance)
        if confiance_moyenne < threshold:
            segments.append([(i*window_size, i*window_size+window_size), confiance_moyenne])

    segments.sort(key=lambda x: x[1])
    segments = [[Segment(segment[0][0]*segmentation.sliding_window.step, segment[0][1]*segmentation.sliding_window.step), segment[1]] for segment in segments]
    segments = segments[:int(len(segments)*annotated_ratio)]
    segments2 = segments
    segments = [x[0] for x in segments]
    segments.sort(key=lambda x: x.start)
    if mode == 'sample' and keep_method == 'lowest':
        segments = segments[:int(len(segments)*annotated_ratio)]
    elif mode == 'sample' and keep_method == 'random':
        segments = np.random.choice(segments, int(len(segments)*annotated_ratio), replace=False)

    with open(file, "a") as f:
        for segment in segments2:
            f.write(sample + ' ' + str(segment[0].start) + ' ' + str(segment[0].end) + ' ' + str(segment[1].item()) + '\n')
    f.close()
    return segments

def alternative_find_low_confidence_frames(sample, segments, threshold, window_size, annotated_ratio, mode, file,keep_method):
    window_steps = int(window_size/segments.sliding_window.step)
    data_wo_nan = segments.data[~np.isnan(segments.data).any(axis=1)]
    sorted = np.sort(data_wo_nan, axis=1)
    confidence = sorted[:, -1] - sorted[:, -2]
    confidence_on_windows = torch.squeeze(avg_pool1d(torch.Tensor([confidence]), window_steps, window_steps, ceil_mode=True))
    indexes = np.where(np.array(confidence_on_windows) <= threshold)[0]
    segments = [Segment(index * window_steps * segments.sliding_window.step, (index+1) * window_steps * segments.sliding_window.step) for index in indexes]
    
    if mode == 'sample' and keep_method == 'lowest':
        segments = segments[:int(len(segments)*annotated_ratio)]
    elif mode == 'sample' and keep_method == 'random':
        segments = np.random.choice(segments, int(len(segments)*annotated_ratio), replace=False)
    

    #create a dico with the segments and the confidence
    segments2 = [[segment, confidence_on_windows[index]] for index, segment in enumerate(segments)]
    segments2.sort(key=lambda x: x[1])
    #drope segments where the confidence is greater than the threshold
    segments2 = [x for x in segments2 if x[1] <= threshold]
    #drop empty segments
    segments2 = [x for x in segments2 if x[0].duration > 0]
    with open(file, "a") as f:
        for segment in segments2:
            f.write(sample + ' ' + str(segment[0].start) + ' ' + str(segment[0].end) + ' ' + str(segment[1].item()) + '\n')
        f.close()
    return segments


def generate_dataset(x_train_file_path, dataset_path, filename, uem_file_name ,pipeline, mode, keep_method, threshold=0.5, window_size=5, annotated_ratio=0.15):
    """ 
        Generate the dataset for the active learning process by calculating the soft segmentation and the low confidence segments for each file in the training set.
        The function also creates a file containing the names of the files to fine tune in the file filename.

        Parameters:
        - x_train_file_path: str
            Path to the file containing the names of the files in the training set.
        - dataset_path: str
            Path to the dataset folder.
        - filename: str
            Name of the file to create containing the names of the files to fine tune.
        - uem_file_name: str
            Name of the file to create containing all the low confidence segments calculated.
        - pipeline: function
            Function to calculate the soft segmentation.
        - mode: str : sample or dataset
            Choose to get annotated_ration % of the lowest confidence frames per sample or per dataset
        - threshold: float, optional
            Confidence threshold, below which frames are considered low confidence.
        - window_size: float, optional
            Size of the sliding window in seconds.
        - annotated_ratio: float, optional
            Ratio of the lowest confidence frames to return.
            """
    assert mode in ['sample', 'dataset'], "mode should be either 'sample' or 'dataset'"
    assert keep_method in ['lowest', 'random'], "keep_method should be either 'lowest' or 'random'"
    if dataset_path.split('/')[-1] == 'ami':
        sufixe ='.Mix-Headset'
    else:
        sufixe = ''

    manual_rttm_path = dataset_path+"/manual_rttm"
    manual_uem_path = dataset_path+"/manual_uems"
    if os.path.exists(manual_rttm_path):
        print(manual_rttm_path," deleted")
        shutil.rmtree(manual_rttm_path)
    os.makedirs(manual_rttm_path)
    if os.path.exists(manual_uem_path):
        print(manual_uem_path," deleted")
        shutil.rmtree(manual_uem_path)
    os.makedirs(manual_uem_path)

    filename = dataset_path+"/lists/"+filename     
    finetune_files = open(filename, "w")
    uem_file = dataset_path+'/'+uem_file_name 
    if os.path.exists(uem_file):
        os.remove(uem_file)
        
    print("Generating soft segmentation and low confidence segments for the fine tuning set")
    size_train = sum(1 for line in open(x_train_file_path))
    with open(x_train_file_path, "r") as x_train:
        for file in tqdm(x_train, total=size_train):
            file = file[:-1]
            wav_file = dataset_path+"/wav/"+file+sufixe+".wav"
            soft_segmentation: segmentation.SlidingWindowFeature = pipeline(wav_file)
            # low_confiance_segments = alternative_find_low_confidence_frames(file, soft_segmentation, threshold, window_size, annotated_ratio, mode, uem_file, keep_method)
            low_confiance_segments = find_low_confiance_frames(file, soft_segmentation, threshold, window_size, annotated_ratio, mode, uem_file, keep_method)
            if len(low_confiance_segments) == 0:
                continue
            if mode =='sample':
                written = write_data_sample(file, low_confiance_segments,dataset_path)
                if written:
                    finetune_files.write(file+'\n')
        
        if mode == 'dataset':
            write_data_dataset(uem_file, keep_method, annotated_ratio, dataset_path,finetune_files)
    print("Fine tuning files created in "+filename)
    finetune_files.close()


def write_data_sample(file, segments,dataset_path):
    annotation = load_rttm(dataset_path+"/rttm/"+file+".rttm")
    _, annotation = annotation.popitem()
    annotated = load_uem(dataset_path+"/uems/"+file+".uem")
    _, annotated = annotated.popitem()

    timeline = Timeline(segments)
    timeline.uri = file                
    new_annotations = annotation.crop(timeline)
    new_annotated = annotated.crop(timeline)
    written = False

    if len(new_annotations) != 0 and len(new_annotated) != 0:

        with open(dataset_path+"/manual_rttm/"+file+".rttm", "w") as rttm:
            new_annotations.write_rttm(rttm)

        with open(dataset_path+"/manual_uems/"+file+".uem", "w") as uem:
            new_annotated.write_uem(uem)

        written = True

    return written

def write_data_dataset(uem_file, keep_method, annotated_ratio, dataset_path,finetune_files):
    uemF = open(uem_file, "r")
    uem = uemF.readlines()
    uem = [x.split() for x in uem]
    uem = [[x[0], float(x[1]), float(x[2]), float(x[3])] for x in uem]
    uem.sort(key=lambda x: x[3])
    if keep_method == 'lowest':
        uem = uem[:int(len(uem)*annotated_ratio)]
    elif keep_method == 'random':
        uem = random.sample( uem, int(len(uem)*annotated_ratio))

    for sample in uem:
        file = sample[0]
        segments = [Segment(sample[1], sample[2])]
        written = write_data_sample(file, segments,dataset_path)
        if written:
            finetune_files.write(file+'\n')
    uemF.close()


