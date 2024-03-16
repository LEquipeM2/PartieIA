import sys
import os
sys.path.append('code/')
import torch
import SoftSeg as segmentation
from ActiveLearning import generate_dataset
import ipywidgets as widgets
from IPython.display import display

def display_choices():
    database_wildget = widgets.RadioButtons(
    options=['AMI', 'Msdwild'],
    value='AMI',
    description='Database:',
    disabled=False,
    )

    widget_generate_new_ds = widgets.RadioButtons(
    options=['Yes', 'No'],
    value='No',
    description='Generate new dataset:',
    disabled=False,
    )
    eval_widget = widgets.RadioButtons(
    options=['Yes', 'No'],
    value='No',
    description='Evaluate the pretrained pipeline :',
    disabled=False,
    )
    widget_validate= widgets.Button(
        description='Validate',
        disabled=False,
        button_style='', 
        tooltip='Validate',
        icon='check'
    )

    def validate(b):
        generate_new_dataset = widget_generate_new_ds.value == "Yes"
        database = database_wildget.value
        if database =="Msdwild":
            x_train_path = "datasets-pyannote/msdwild/lists/custom1_train.txt"
            dataset_path = "datasets-pyannote/msdwild" 
        else:
            x_train_path = "datasets-pyannote/ami/lists/train.mini.txt"
            dataset_path = "datasets-pyannote/ami"
        widget_mode, widget_method, widget_annotated_ratio, widget_windows, widget_generete_ds = ds_choices(generate_new_dataset, x_train_path, dataset_path)
        display(widget_mode, widget_method, widget_annotated_ratio, widget_windows, widget_generete_ds)
        
    
    widget_validate.on_click(validate)
    return database_wildget, widget_generate_new_ds, eval_widget, widget_validate


def ds_choices(generate_new_dataset, x_train_path, dataset_path ):
    model_seg = "pyannote/segmentation-3.0"
    HF_TOKEN = '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'
    model_seg = "pyannote/segmentation-3.0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = segmentation.SoftSpeakerSegmentation(segmentation=model_seg, use_auth_token=HF_TOKEN)
    pipeline.to(device)

    
    if generate_new_dataset:
        widget_mode = widgets.RadioButtons(
            options=['sample', 'dataset'],
            value='sample',
            description='Mode:',
            disabled=False
        )
        widget_method = widgets.RadioButtons(
            options=['random', 'lowest'],
            value='lowest',
            description='Method:',
            disabled=False
        )
        widget_annotated_ratio = widgets.FloatText(
            value=0.3,
            step =0.1,
            description='Annotated ratio :',
            disabled=False,
            adaptive_height=True,
            layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
            style={"description_width": "initial"}
        )

        widget_windows = widgets.FloatText(
            value=7.5,
            step =0.5,
            description='Size of the sliding window (seconds):',
            disabled=False,
            adaptive_height=True,
            layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
            style={"description_width": "initial"}
        )

        widget_generete_ds = widgets.Button(
            description='Generate dataset',
            disabled=False,
            button_style='', 
            tooltip='Generate dataset',
            icon='check'
        )

        def launch_generate_dataset(b):
            mode_value = widget_mode.value
            method_value = widget_method.value
            annotated_ratio_value = widget_annotated_ratio.value
            windows_value = widget_windows.value
            generate_dataset(x_train_path,dataset_path,"fine_uem.txt" ,"alltimelines.uem" ,pipeline,mode=mode_value, keep_method=method_value,window_size=windows_value,annotated_ratio=annotated_ratio_value)

        widget_generete_ds.on_click(launch_generate_dataset)
        return widget_mode, widget_method, widget_annotated_ratio, widget_windows, widget_generete_ds