import sys
import os
sys.path.append('code/')
from ActiveLearning import *
import SoftSeg as segmentation
import torch
import pyannotebook_reborn
from IPython.display import clear_output, display
import ipywidgets as widgets
from tqdm.notebook import tqdm_notebook as tqdm
import pickle

def annotate_sample(dataset_path,fine_tuned_file, num_samples_processed,iter_samples,dico_samples):

    num_samples_processed += 1 
    num_samples_remaining = len(dico_samples) - num_samples_processed 
    next_item = next(iter_samples) 
    
    widget = pyannotebook_reborn.Pyannotebook(dataset_path+"/wav/"+next_item+".wav")
    timeline = dico_samples[next_item]
    if type(timeline) != list:
        timeline = timeline.tolist()
    widget.timelines = timeline
    print(f"Please annotate the low confidence segment(s) shown and click on the save button to save the annotation for the file {next_item}\n{num_samples_remaining} samples remaining to annotate.")
    display(widget)
    
    button = widgets.Button(description="Save")
    display(button)
    
    def on_button_clicked(b, next_item,widget,fine_tuned_file):
        fine_tuned_file.write(next_item+'\n')
        with open("manual_annotations/" + next_item+ ".rttm", "w") as rttm:
            widget.annotation.write_rttm(rttm)
        
        with open("manual_annotations/"+next_item+".uem", "w") as uem:
            widget.annotation.get_timeline().write_uem(uem)
        display("Annotations and Timeline Saved in manual_annotations folder for " + next_item+ "\n")
        rttm.close()
        uem.close()
        
        if num_samples_remaining > 0:
            # next_item = next(iter_samples)
            clear_output() 
            # widget = pyannotebook_reborn.Pyannotebook(dataset_path+"/wav/"+next_item+".wav")
            annotate_sample(dataset_path,fine_tuned_file,num_samples_processed,iter_samples,dico_samples)
            
        else:
            fine_tuned_file.close()
            clear_output() 
            print("All samples have been annotated.")

    button.on_click(lambda b: on_button_clicked(b, next_item,widget,fine_tuned_file))

def manual_annotation(dataset_path, xtrain_file_name, fine_tuned_file, HF_token,mode,method, generate_set=True ,annotated_ratio=0.3, window_size=7.5, threshold=0.5, suffixe=""):
    #Etape 1 : faire la segmentation et calcul low confidence segments
    #Dictionnaire qui contiendra les segments de basse confiance
    dico_samples = {}
    #Faire la segmentation des fichiers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = segmentation.SoftSpeakerSegmentation(segmentation="pyannote/segmentation-3.0", use_auth_token=HF_token)
    pipeline.to(device)
    if generate_set :
        #Récupérer les fichiers à traiter
        path_data_file = dataset_path+'/lists/'+xtrain_file_name
        with open(path_data_file, "r") as f:
            train_data = f.readlines()
        data = [x.strip()+suffixe for x in train_data]
        f.close()
        print("Generating soft segmentation and low confidence segments for the fine tuning set.\n")
        for sample in tqdm(data):
            wav_path = dataset_path+'/wav/'+sample+'.wav'
            soft_segmentation: segmentation.SlidingWindowFeature = pipeline(wav_path)
            low_conf_seg = find_low_confiance_frames(sample,soft_segmentation, threshold, window_size, annotated_ratio,mode,dataset_path+'/lists/all_timelines.uem',method)
            dico_samples[sample] = low_conf_seg
        #Sauvegarde le dictionnaire pour réutilisation
        with open("low_conf_dico.pkl", "wb") as f:
            pickle.dump(dico_samples, f)
        f.close()

    else:
        dico_samples = pickle.load(open('low_conf_dico.pkl', 'rb'))
        print("Low confidence segments loaded.\n")

    #Etape 2 : Annotation manuelle des samples
    #Les nouvelles annotations seront enregistrées dans le dossier manual_annotations (rttm et uem)
    if not os.path.exists("manual_annotations"):
        os.makedirs("manual_annotations")
    iter_samples = iter(dico_samples)
    num_samples_processed = 0
    finetune_files = open('manual_annotations/'+fine_tuned_file, "w")
    annotate_sample(dataset_path,finetune_files, num_samples_processed,iter_samples,dico_samples)
