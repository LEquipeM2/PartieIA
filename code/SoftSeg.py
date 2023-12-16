# Code mis à disposition par Alexis PLAQUET et repris pour le projet

import functools
from typing import Callable, Optional, Text, Union

import numpy as np
import torch
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_model,
)
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.permutation import permutate


class SoftSpeakerSegmentation(Pipeline):
    """Speaker segmentation pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation-3.0".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    segmentation_step: float, optional
        The segmentation model is applied on a window sliding over the whole audio file.
        `segmentation_step` controls the step of this window, provided as a ratio of its
        duration. Defaults to one third (i.e. 66% overlap between two consecutive windows).
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 1.
    use_auth_token : str, optional
        When loading private huggingface.co models, set `use_auth_token`
        to True or to a string containing your hugginface.co authentication
        token that can be obtained by running `huggingface-cli login`

    Usage
    -----
    # perform speaker segmentation
    >>> pipeline = SpeakerSegmentation()
    >>> segmentation: SlidingWindowFeature = pipeline("/path/to/audio.wav")

    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation-3.0",
        segmentation_step: float = 1 / 3,
        segmentation_batch_size: int = 1,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__()

        self.segmentation_model = segmentation
        model: Model = get_model(segmentation, use_auth_token=use_auth_token)

        specifications = model.specifications
        if not specifications.powerset:
            raise ValueError("Only powerset segmentation models are supported.")

        self.segmentation_step = segmentation_step

        segmentation_duration = model.specifications.duration
        self._segmentation = Inference(
            model,
            duration=segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
            skip_conversion=True,
            batch_size=segmentation_batch_size,
        )
        self._frames: SlidingWindow = self._segmentation.model.example_output.frames

        self._powerset = Powerset(
            len(specifications.classes), specifications.powerset_max_classes
        )

    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    # Should be added to the Powerset class eventually
    def get_permutated_ps(self, ps: Powerset, t_ps: torch.Tensor, permutation: torch.Tensor):
        mapping = ps.mapping
        permutated_mapping = mapping[:, permutation]

        # create mapping-shaped 2**N tensor
        arange = torch.arange(mapping.shape[1], device=mapping.device, dtype=torch.int)
        powers2 = (2**arange).tile((ps.mapping.shape[0], 1))

        indexing_og = torch.sum(mapping * powers2, dim=-1).long()
        indexing_new = torch.sum(permutated_mapping * powers2, dim=-1).long()

        ps_permutation = (indexing_og[None] == indexing_new[:, None]).int().argmax(dim=0)

        return t_ps[..., ps_permutation]

    def get_segmentations(self, file, hook=None) -> SlidingWindowFeature:
        """Apply segmentation model

        Parameter
        ---------
        file : AudioFile
        hook : Optional[Callable]

        Returns
        -------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        """

        if hook is None:
            inference_hook = None
        else:
            inference_hook = functools.partial(hook, "segmentation", None)

        if self.training:
            segmentations = file.setdefault("cache", dict()).setdefault(
                "segmentation", None
            )

            if segmentations is None:
                segmentations = self._segmentation(file, hook=inference_hook)
                file["cache"]["segmentation"] = segmentations

            return segmentations

        return self._segmentation(file, hook=inference_hook)

    def apply(
        self,
        file: AudioFile,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        segmentation : SlidingWindowFeature
            Speaker segmentation
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model on a sliding window
        powerset_segmentations = self.get_segmentations(file, hook=hook)
        hook("powerset_segmentation", powerset_segmentations)
        # shape: (num_chunks, num_frames, local_num_speakers)

        num_chunks, num_frames, _ = powerset_segmentations.data.shape

        # convert from powerset to multilabel segmentation
        multilabel_segmentations = SlidingWindowFeature(
            self._powerset.to_multilabel(
                torch.tensor(powerset_segmentations.data),
                soft=True,
            ).numpy(force=True),
            powerset_segmentations.sliding_window,
        )

        permutated_segmentations = np.zeros_like(multilabel_segmentations.data)
        permutated_ps_segmentations = np.zeros_like(powerset_segmentations.data)

        # number of frames in common between two consecutive chunks
        num_overlapping_frames = round((1 - self.segmentation_step) * num_frames)

        # permutate each window to match previous one as much as possible
        for c, ((_, segmentation), (_, ps_segmentation)) in enumerate(zip(multilabel_segmentations, powerset_segmentations)):
            hook("permutated_segmentation", None, completed=c, total=num_chunks)

            if c == 0:
                previous_segmentation = segmentation
                previous_segmentation_ps = ps_segmentation 
            else:
                permutation = permutate(
                    previous_segmentation[-num_overlapping_frames:][None],
                    segmentation[:num_overlapping_frames],
                )[1][0]
                previous_segmentation = segmentation[:, permutation]
                previous_segmentation_ps = self.get_permutated_ps(self._powerset, ps_segmentation, permutation)

            permutated_segmentations[c] = previous_segmentation
            permutated_ps_segmentations[c] = previous_segmentation_ps

        permutated_segmentations = SlidingWindowFeature(
            permutated_segmentations, multilabel_segmentations.sliding_window
        )
        permutated_ps_segmentations = SlidingWindowFeature(
            permutated_ps_segmentations, powerset_segmentations.sliding_window
        )

        hook(
            "permutated_segmentation",
            permutated_segmentations,
            completed=num_chunks,
            total=num_chunks,
        )

        hook(
            "permutated_ps_segmentation",
            permutated_ps_segmentations,
            completed=num_chunks,
            total=num_chunks,
        )

        aggregated_ps_segmentations = Inference.aggregate(
                permutated_ps_segmentations, self._frames, hamming=False, skip_average=False
        )
        hook(
            "aggregated_ps_segmentation",
            aggregated_ps_segmentations,
            completed=num_chunks,
            total=num_chunks,
        )

        return Inference.aggregate(
            permutated_segmentations, self._frames, hamming=False, skip_average=False
        )