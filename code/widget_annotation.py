from IPython.display import clear_output, display
import sys
sys.path.append('code/')
import ipywidgets as widgets
from annotations import *

def launch_annotation_process():
    param_buttons()


def create_buttons():
    #widget for the annotated ratio window size and threshold
    widget_text_file = widgets.Text(
    placeholder='Ex : train_set.txt ',
    description='Name of the file containing the list of train samples :',
    disabled=False,
        adaptive_height=True,
        layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
        style={"description_width": "initial"}
    )

    annotated_ratio_widget = widgets.FloatText(
        value=0.3,
        step=0.1,
        max=1,
        min=0,
        description='Annotated Ratio (float):',
        disabled=False,
        adaptive_height=True,
        layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
        style={"description_width": "initial"}

    )

    window_size_widget = widgets.FloatText(
    value=7.5,
    description='Window Size in seconds :',
    disabled=False,
    adaptive_height=True,
    layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
    style={"description_width": "initial"}
    )

    threshold_widget = widgets.FloatText(
    value=0.5,
    description='Threshold of the confidence (float) :',
    disabled=False,
    adaptive_height=True,
    layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
    style={"description_width": "initial"}
    )

    mode_widget = widgets.RadioButtons(
    options=['dataset', 'sample'],
    value='sample',
    description='Apply annotated ratio on the full dataset or on each sample ?',
    disabled=False,
    layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
    style={"description_width": "initial"}
    )

    method_widget = widgets.RadioButtons(
    options=['random', 'lowest'],
    value='lowest',
    description='Method to select the samples to annotate :',
    disabled=False,
    layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
    style={"description_width": "initial"}
    )

    return annotated_ratio_widget, window_size_widget, threshold_widget,widget_text_file, mode_widget, method_widget


def create_generate_set_button():
   #choix generer ou non nouveay set
    generate_set = widgets.RadioButtons(
      options=['Yes', 'No'],
      value='No',
      description='Generate a new fine tuned set ?',
      disabled=False
   ) 
    generate_set_bool = False
    annotated_ratio, window_size, threshold = 0.3, 7.5, 0.5
    xtrain_file = None
    mode = 'sample'
    method = 'lowest'
    
    def on_value_change(change):
      if change.new == "Yes":
        clear_output(wait=True)
        annotated_ratio_widget, window_size_widget, threshold_widget, widget_text_file,mode_widget, method_widget = create_buttons()
        # annotated_ratio, window_size, threshold, xtrain_file = annotated_ratio_widget.value, window_size_widget.value, threshold_widget.value, widget_text_file.value
        # generate_set_bool = True
        validate_button1 = widgets.Button(description="Validate",
                                        layout=widgets.Layout(display="flex",
                                                    flex_flow="column",
                                                    align_items="flex-start",
                                                    width="auto", height="auto"))
        
        def on_button_clicked(b):
            annotated_ratio, window_size, threshold, xtrain_file = annotated_ratio_widget.value, window_size_widget.value, threshold_widget.value, widget_text_file.value
            generate_set_bool = True
            mode, method = mode_widget.value, method_widget.value
            # validate_button1 = widgets.Button(description="Validate",
            #                                 layout=widgets.Layout(display="flex",
            #                                             flex_flow="column",
            #                                             align_items="flex-start",
            #                                             width="auto", height="auto"))
        
            create_manual_annotation_button(annotated_ratio, window_size, threshold, generate_set_bool, suffixe, xtrain_file, mode, method)

        validate_button1.on_click(on_button_clicked)

        # param_buttons()
        display(generate_set, widget_text_file, annotated_ratio_widget, window_size_widget, threshold_widget,mode_widget,method_widget ,validate_button1)
      else:
        clear_output(wait=True)
        # param_buttons()
        # generate_set_bool = False
        # annotated_ratio, window_size, threshold = 0.3, 7.5, 0.5
        display(generate_set)


    generate_set.observe(on_value_change, 'value')
    display(generate_set)

    validate_button = widgets.Button(description="Validate",
                                     layout=widgets.Layout(display="flex",
                                                 flex_flow="column", 
                                                 align_items="flex-start", 
                                                 width="auto", height="auto"))

    def on_button_clicked(b):
        
        create_manual_annotation_button(annotated_ratio, window_size, threshold, generate_set_bool, suffixe, xtrain_file, mode, method)

    validate_button.on_click(on_button_clicked)
    display(validate_button)


def param_buttons():
    widget_text_dataset = widgets.Text(
    value='datasets-pyannote/ami',
    placeholder='Path to the dataset',
    description='Dataset path :',
    disabled=False,
        adaptive_height=True,
        layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
        style={"description_width": "initial"}
    )

    widget_text_fine = widgets.Text(
    placeholder='Ex : finetune_set.txt ',
    description='Name of the file that will contains the fine tune samples :',
    disabled=False,
        adaptive_height=True,
        layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
        style={"description_width": "initial"}
    )

    #widget text for the HF token
    widget_text_token = widgets.Text(
    value='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',
    placeholder='HF token',
    description='HF token:',
    disabled=False,
        adaptive_height=True,
        layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
        style={"description_width": "initial"}
    )

    #widget text for the suffixe
    widget_text_suffixe = widgets.Text(
    value='.Mix-Headset',
    placeholder='Suffixe',
    description='Suffixe of audio (exemple : .Mix-Headset for AMI):',
    disabled=False,
        adaptive_height=True,
        layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start", width="auto", height="auto"),
        style={"description_width": "initial"}
    )

    #button validate
    button = widgets.Button(description="Validate", 
                            layout=widgets.Layout(display="flex",
                                                 flex_flow="column", 
                                                 align_items="flex-start", 
                                                 width="auto", height="auto"))
    
    def on_button_clicked(b):
        global dataset_path,  HF_token, suffixe, fine_tuned_file
        dataset_path = widget_text_dataset.value
        HF_token = widget_text_token.value
        suffixe = widget_text_suffixe.value
        fine_tuned_file = widget_text_fine.value
        clear_output(wait=True)
        display(create_generate_set_button())

    button.on_click(on_button_clicked)
    display(widget_text_dataset, widget_text_fine ,widget_text_token, widget_text_suffixe, button)



def create_manual_annotation_button(annotated_ratio, window_size, threshold,generate_set_bool, suffixe, xtrain_file, mode, method):
    button = widgets.Button(description="Launch Manual Annotation", 
                            layout=widgets.Layout(display="flex",
                                                 flex_flow="column", 
                                                 align_items="flex-start", 
                                                 width="auto", height="auto"))

    def on_button_clicked(b):
        manual_annotation(dataset_path, xtrain_file,fine_tuned_file, HF_token, mode,method,generate_set_bool, annotated_ratio, window_size, threshold, suffixe)
    button.on_click(on_button_clicked)
    display(button)

