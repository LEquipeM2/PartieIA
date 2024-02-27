import os
import SoftSeg as segmentation
from pyannote.core import Annotation
from tqdm.notebook import trange, tqdm
from pyannote.database.util import load_rttm, load_uem
from pyannote.core import Segment
import numpy as np


def find_low_confiance_frames(segmentation, threshold, window_size, annotated_ratio):
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
    segments = [x[0] for x in segments]
    segments.sort(key=lambda x: x.start)
    return segments


def generate_dataset(x_train_file_path, dataset_path, filename ,pipeline,threshold=0.5, window_size=5, annotated_ratio=0.15):
    """ 
        Generate the dataset for the active learning process by calculating the soft segmentation and the low confidence segments for each file in the training set.
        The function also creates a file containing the names of the files to fine tune in the file filename.

        Parameters:
        - x_train_file_path: str
            Path to the file containing the names of the files in the training set.
            - dataset_path: str
            Path to the dataset folder.
            - pipeline: function
            Function to calculate the soft segmentation.
            - threshold: float, optional
            Confidence threshold, below which frames are considered low confidence.
            - window_size: float, optional
            Size of the sliding window in seconds.
            - annotated_ratio: float, optional
            Ratio of the lowest confidence frames to return.
            """
    if dataset_path.split('/')[-1] == 'ami':
        sufixe ='.Mix-Headset'
    else:
        sufixe = ''
    finetune_files = open(filename, "w")
    print("Generating soft segmentation and low confidence segments for the fine tuning set")
    with open(x_train_file_path, "r") as x_train:
        for file in tqdm(x_train):
            file = file[:-1]
            wav_file = dataset_path+"/wav/"+file+sufixe+".wav"
            soft_segmentation: segmentation.SlidingWindowFeature = pipeline(wav_file)
            low_confiance_segments = find_low_confiance_frames(soft_segmentation, threshold, window_size, annotated_ratio)
            annotation = load_rttm(dataset_path+"/rttm/"+file+".rttm")
            _,annotation =  annotation.popitem()
            annotated = load_uem(dataset_path+"/uems/"+file+".uem")
            _,annotated = annotated.popitem()

            if len(low_confiance_segments) == 0:
                continue
            start = low_confiance_segments[0].start
            end = low_confiance_segments[-1].end
            new_annotation = Annotation()
            new_annotation = annotation.crop(Segment(start,end))
            annotated = annotated.crop(Segment(start,end))
            
            if not os.path.exists(dataset_path+"/manual_rttm"):
                os.makedirs(dataset_path+"/manual_rttm")
            with open(dataset_path+"/manual_rttm/"+file+".rttm", "w") as rttm:
                new_annotation.write_rttm(rttm)
            rttm.close()
            if not os.path.exists(dataset_path+"/manual_uems"):
                os.makedirs(dataset_path+"/manual_uems")
            with open(dataset_path+"/manual_uems/"+file+".uem", "w") as uem:
                annotated.write_uem(uem)
            uem.close()
            finetune_files.write(file+'\n')
    finetune_files.close()
    print("Fine tuning files created in "+filename)

