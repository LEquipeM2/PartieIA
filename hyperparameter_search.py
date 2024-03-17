import sys
import os
sys.path.append('code/')
import numpy as np
import torch
import matplotlib.pyplot as plt
import SoftSeg as segmentation
from SoftSeg import find_low_confiance_frames
import wave
from pyannote.core import Annotation,Timeline
from tqdm.notebook import trange
from pyannote.database.util import load_rttm, load_uem
from pyannote.core import Segment
from pyannote.audio.pipelines.utils.hook import Hooks, ArtifactHook, TimingHook, ProgressHook
from pyannote.audio import Pipeline, Inference
from pyannote.audio import Model
from pyannote.database import registry
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm
import pickle
from ActiveLearning import generate_dataset

HF_TOKEN = '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'
model_seg = "pyannote/segmentation-3.0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

database ="msdwild"

if database == "ami":
    x_train_path = "datasets-pyannote/ami/lists/train.mini.txt"
    dataset_path = "datasets-pyannote/ami"
elif database == "msdwild":
    x_train_path = "datasets-pyannote/msdwild/lists/custom1_train.txt"
    dataset_path = "datasets-pyannote/msdwild"

evaluate = True

params_windowSize = [30, 35, 40, 45, 50, 60]
params_annotatedRatio = [1]
powerset = []
for ws in params_windowSize:
    for ar in params_annotatedRatio:
        powerset.append((ws, ar))
print(powerset)
combos_performances = []
best_combo = powerset[0]
best_DER = 1

#Ne pas oublier de changer le fichier train dans database.yml pour qu'il pointe vers le bon fichier de fine tuning
if database == "ami":
    protocol = "AMI.SpeakerDiarization.mini"
    yaml_path = "datasets-pyannote/ami/pyannote/database.yml"
elif database == "msdwild":
    protocol = "MSDWILD.SpeakerDiarization.CustomFew"
    yaml_path = "datasets-pyannote/msdwild/database.yml"

registry.load_database(yaml_path)
dataset = registry.get_protocol(protocol)
print("Checking that the 'annotation' key is present in all train files...")
for file in dataset.train():
    assert "annotation" in file
print("Checking that the 'annotation' key is present in all test files...")
for file in dataset.test():
    assert "annotation" in file

pretrained_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
pretrained_pipeline.to(torch.device(device))

torch.cuda.empty_cache()
metric_pretrained = DiarizationErrorRate()
if evaluate:
    for file in tqdm(dataset.test()):
        if file["database"] == "AMI":
            path_to_wav = "datasets-pyannote/ami/wav/"
            suffixe = ".Mix-Headset"
        elif file["database"] == "MSDWILD":
            path_to_wav = "datasets-pyannote/msdwild/wav/"
            suffixe = ""
        file["pretrained pipeline"] = pretrained_pipeline(path_to_wav+file["uri"]+suffixe+".wav")
        metric_pretrained(file["annotation"], file["pretrained pipeline"], uem=file["annotated"], detailed=True)
    print(f"\nThe pretrained pipeline reaches a Diarization Error Rate (DER) of {100 * abs(metric_pretrained):.1f}% on test set.")

for combo in powerset:

    pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=HF_TOKEN)
    pipeline = segmentation.SoftSpeakerSegmentation(segmentation=model_seg, use_auth_token=HF_TOKEN)
    pipeline.to(device)

    generate_dataset(x_train_path, dataset_path, dataset_path+"/lists/fine_uem.txt", pipeline, window_size=combo[0], annotated_ratio=combo[1])

    #Ne pas oublier de changer le fichier train dans database.yml pour qu'il pointe vers le bon fichier de fine tuning
    if database == "ami":
        protocol = "AMI.SpeakerDiarization.mini"
        yaml_path = "datasets-pyannote/ami/pyannote/database.yml"
    elif database == "msdwild":
        protocol = "MSDWILD.SpeakerDiarization.CustomFew"
        yaml_path = "datasets-pyannote/msdwild/database.yml"

    registry.load_database(yaml_path)
    dataset = registry.get_protocol(protocol)
    print("Checking that the 'annotation' key is present in all train files...")
    for file in dataset.train():
        assert "annotation" in file
    print("Checking that the 'annotation' key is present in all test files...")
    for file in dataset.test():
        assert "annotation" in file

    
    pretrained_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    pretrained_pipeline.to(torch.device(device))

    from types import MethodType
    from torch.optim import Adam
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        RichProgressBar,
    )


    from pyannote.audio.tasks import Segmentation
    model = Model.from_pretrained(model_seg, use_auth_token=HF_TOKEN)
    task = Segmentation(
        dataset,
        duration=model.specifications.duration,
        max_num_speakers=len(model.specifications.classes),
        batch_size=32,
        num_workers=2,
        loss="bce",
        vad_loss="bce")
    model.task = task
    model.setup(stage="fit")


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    model.configure_optimizers = MethodType(configure_optimizers, model)

    monitor, direction = task.val_monitor
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_top_k=1,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=False,
        filename="{epoch}",
        verbose=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=direction,
        min_delta=0.0,
        patience=10,
        strict=True,
        verbose=False,
    )

    callbacks = [RichProgressBar(), checkpoint, early_stopping]

    from pytorch_lightning import Trainer
    trainer = Trainer(accelerator="gpu",
                    callbacks=callbacks,
                    max_epochs=20,
                    gradient_clip_val=0.5)
                    
    trainer.fit(model)


    finetuned_model = checkpoint.best_model_path
    #on a entrainé sur google colab et récupéré le modèle
    # finetuned_model = Model.from_pretrained('epoch=17.ckpt')
    with open("hyperparameters.pickle", 'rb') as handle:
        hparameters = pickle.load(handle)

    from pyannote.audio.pipelines import SpeakerDiarization
    finetuned_pipeline = SpeakerDiarization(
        segmentation=finetuned_model,
        embedding=pretrained_pipeline.embedding,
        embedding_exclude_overlap=pretrained_pipeline.embedding_exclude_overlap,
        clustering=pretrained_pipeline.klustering,
    )

    finetuned_pipeline.to(device)

    finetuned_pipeline.instantiate({
        "segmentation": {
            "threshold": hparameters['best_segmentation_threshold'],
            "min_duration_off": 0.0,
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 12,
            "threshold": hparameters['best_clustering_threshold'],
        },
    })

    metric_finetuned = DiarizationErrorRate()
    torch.cuda.empty_cache()
    for file in tqdm(dataset.test()):
        file["finetuned pipeline"]  = finetuned_pipeline(path_to_wav+file["uri"]+suffixe+".wav")
        metric_finetuned(file["annotation"], file["finetuned pipeline"], uem=file["annotated"], detailed=True)
    print(f"The finetuned pipeline reaches a Diarization Error Rate (DER) of {100 * abs(metric_finetuned):.1f}% on {database} test set.")
    print(f"Combo used:", combo)
    combos_performances.append((combo, metric_finetuned[:]))
    if abs(metric_finetuned) < best_DER:
        best_DER = abs(metric_finetuned)
        best_combo = combo

print(f"best DER ({best_DER}) was achieved with window size {best_combo[0]} and annotation rate {best_combo[1]}")
print(combos_performances)