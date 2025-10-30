# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) 2018 Ryan Leary
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# This file contains code artifacts adapted from https://github.com/ryanleary/patter
import copy
import inspect
import io
import math
import os
import random
from abc import ABC
from torch.distributions.uniform import Uniform

from typing import  List, Optional, Tuple, Type, Union, cast, overload

import numpy as np
import json
import torch
import torch.nn as nn
from pathlib import Path
import torchaudio
import torch.nn.functional as F

from registrable import Registrable
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import speed

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import IterableDataset
from nemo.utils import logging

from .augmentation_utils import *


# TODO @blisc: Perhaps refactor instead of import guarding
HAVE_OMEGACONG_WEBDATASET = True
try:
    import webdataset as wds
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:
    from nemo.utils.exceptions import LightningNotInstalledException

    HAVE_OMEGACONG_WEBDATASET = False


try:
    from nemo.collections.asr.parts.utils import numba_utils

    HAVE_NUMBA = True
except (ImportError, ModuleNotFoundError):
    HAVE_NUMBA = False


def read_one_audiosegment(manifest, target_sr, tarred_audio=False, audio_dataset=None):
    if tarred_audio:
        if audio_dataset is None:
            raise TypeError("Expected augmentation dataset but got None")
        audio_file, file_id, manifest_entry = next(audio_dataset)

        offset = 0 if manifest_entry.offset is None else manifest_entry.offset
        duration = 0 if manifest_entry.duration is None else manifest_entry.duration

    else:
        audio_record = random.sample(manifest.data, 1)[0]
        audio_file = audio_record.audio_file
        offset = 0 if audio_record.offset is None else audio_record.offset
        duration = 0 if audio_record.duration is None else audio_record.duration

    return AudioSegment.from_file(audio_file, target_sr=target_sr, offset=offset, duration=duration)


class Augmentation(ABC, Registrable, nn.Module):
    """
    Base class for all effects that can be used for data augmentation

    Attributes
    ----------
    cpu_only: `bool`
        Denotes if an effect can only operate on CPU
    is_batch_agnostic: `bool`
        Denotes if the SAME effect is applied to the entire batch of signals.
        One of the cases where it's false is when the batch of signals are iterated over to apply the different
        config of effect for each item.
        This value is True if all the signals within the same batch undergo same transformation.
    """

    def __init__(self, cpu_only: bool, is_batch_agnostic: bool):
        super().__init__()
        self.cpu_only = cpu_only
        self.is_batch_agnostic = is_batch_agnostic

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @overload
    def forward(
        self, input_tensor: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def forward(self, input_tensor: np.ndarray, input_lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

    def forward(
        self,
        input_tensor: Union[np.ndarray, torch.Tensor],
        input_lengths: Union[np.ndarray, torch.Tensor],
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        is_numpy_input = isinstance(input_tensor, np.ndarray) and isinstance(input_lengths, np.ndarray)

        if is_numpy_input:
            input_tensor = torch.from_numpy(input_tensor)
            input_lengths = torch.from_numpy(input_lengths)

        input_tensor = cast(torch.Tensor, input_tensor)
        input_lengths = cast(torch.Tensor, input_lengths)

        output_tensor, output_lengths = self.transform(input_tensor, input_lengths)

        if is_numpy_input:
            output_tensor = output_tensor.detach().cpu().numpy()
            output_lengths = output_lengths.detach().cpu().numpy()

        return output_tensor, output_lengths


####### Custom Augmentations #######


class SpeedPerturbation(Augmentation):
    """
    Changes the speed of the input audio signal. Uses `torchaudio.functional.speed` function
    underneath.

    Attributes
    ----------
    min_speed: `float` ( default = 0.9 )
        Minimum speed factor to be applied
    max_speed: `float` ( default = 1.1 )
        Maximum speed factor to be applied
    original_freq: `int` ( default = DEFAULT_SAMPLE_RATE )
        Original frequency of the input signal
    is_batch_agnostic: `bool` ( default = True )
        If True, the effect is applied to the entire batch at once, else applied individually
        for each item in the batch.
    """

    def __init__(
        self,
        min_speed=0.9,
        max_speed=1.1,
        original_freq=DEFAULT_SAMPLE_RATE,
        is_batch_agnostic=True,
    ) -> None:

        super().__init__(cpu_only=False, is_batch_agnostic=is_batch_agnostic)

        self.min_speed = min_speed
        self.max_speed = max_speed
        self.original_freq = original_freq

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.is_batch_agnostic:
            output_tensor, output_lengths = speed(
                input_tensor,
                orig_freq=self.original_freq,
                factor=round(random.uniform(self.min_speed, self.max_speed), 1),
                lengths=input_lengths,
            )
            # print(f"max value: {output_tensor.max()} min value: {output_tensor.min()}")
        else:
            outputs, output_lengths = zip(
                *[
                    speed(
                        input_tensor[i],
                        orig_freq=self.original_freq,
                        factor=round(random.uniform(self.min_speed, self.max_speed), 1),
                        lengths=input_lengths[i],
                    )
                    for i in range(input_tensor.shape[0])
                ]
            )
            output_tensor = pad_sequence(outputs, batch_first=True).to(input_tensor.device)
            output_lengths = torch.tensor(output_lengths, dtype=torch.float32, device=input_tensor.device)

        return output_tensor, output_lengths


class Gain(Augmentation):
    def __init__(self, min_db=-30, max_db=-6, min_segments=2, max_segments=6, sample_rate=16000, fade=True):
        """
        Simulates volume variations in a telephone conversation.

        Attributes
        ----------
        min_db: `float` ( default = -30 )
            Minimum gain in dB
        max_db: `float` ( default = -6 )
            Maximum gain in dB
        min_segments: `int` ( default = 2 )
            Minimum number of gain segments
        max_segments: `int` ( default = 6 )
            Maximum number of gain segments
        sample_rate: `int` ( default = 16000 )
            Sample rate of the audio signal
        fade: `bool` ( default = True )
            If True, apply fade in/out effect to the segments
        """
        super().__init__(False, True)
        self.min_db = min_db
        self.max_db = max_db
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.sample_rate = sample_rate
        self.fade = fade

    def transform(self, audio: torch.Tensor, audio_signal_len: torch.Tensor) -> torch.Tensor:
        _, total_samples = audio.shape
        device = audio.device

        # Select number of segments
        num_segments = torch.randint(self.min_segments, self.max_segments + 1, (1,)).item()*2
        
        segment_starts = torch.sort(
            torch.randint(1, total_samples - 1, (num_segments - 1,), device=device)
        ).values.tolist()

        segment_starts = [0] + segment_starts + [total_samples]

        # Convert dB to gain (scaling factor)
        gains_db = torch.empty(num_segments, device=device).uniform_(self.min_db, self.max_db)
        gains = 10 ** (gains_db / 20)

        augmented_audio = audio.clone()
        prev_end = 0
        for i in range(num_segments):
            start, end = segment_starts[i], segment_starts[i + 1]

            if torch.rand(1, device=device) < 0.5:  # Randomly apply to some segments
                gain = gains[i]
                if self.fade and prev_end!=start and prev_end!=0:  # Apply fade in only if the segments are not adjacent and not the first segment
                    fade_length = min((end - start) // 4, self.sample_rate // 20)  # ~50ms fade
                    fade = torch.linspace(1, gain, fade_length, device=device)
                    augmented_audio[:, start:start + fade_length] *= fade
                    augmented_audio[:, start + fade_length:end - fade_length] *= gain
                else:
                    augmented_audio[:, start:end] *= gain
                prev_end = end

        return augmented_audio, audio_signal_len

class TimeStretch(Augmentation):
    """
    Changes the tempo of the input audio signal. Uses `torchaudio.functional.phase_vocoder`
    function underneath.

    Attributes
    ----------
    stretch_factor: `Optional[float]` ( default = None )
        Tempo of the input will be changed by this factor
    stretch_factor_range: `Tuple[float]` ( default = (0.75, 1, 1.25, 1.5, 1.75, 2) )
        To randomize the stretch factor used during every call, range of
        values can be provided. At every call a random value is chosen from this range.
    stft_win_length: `int` ( default = 320 )
        Window length for the the stft/istft that is applied during this transformation.
    stft_hop_length: `int` ( default = 160 )
        Hop length for the the stft/istft that is applied during this transformation.
    stft_n_fft: `int` ( default = 512 )
        n_fft for the the stft/istft that is applied during this transformation.
    """

    def __init__(
        self,
        stretch_factor: Optional[float] = None,
        stretch_factor_range: Tuple[float, ...] = (0.85,1.15),
        stft_win_length: int = 320,
        stft_hop_length: int = 160,
        stft_nfft: int = 512,
    ):
        # This is not batch_agnostic effect because we are generating a random stretch factor for every item.
        super().__init__(cpu_only=False, is_batch_agnostic=False)

        if stretch_factor is None and len(stretch_factor_range) == 0:
            raise ValueError(
                "`stretch_factor` must have a value, else provide a valid set of values for `stretch_factor_range`"
            )
        if len(stretch_factor_range)>2:
            raise ValueError("`stretch_factor_range` should have at most 2 values")
        
        self.stretch_factor = stretch_factor
        self.stretch_factor_range = stretch_factor_range

        self.stft_win_length = stft_win_length
        self.stft_hop_length = stft_hop_length
        self.stft_nfft = stft_nfft
        self.num_freq = self.stft_nfft // 2 + 1

    def transform(
        self, input_tensor: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batched_input = len(input_tensor.shape) == 2
        if not batched_input:
            input_tensor = input_tensor.unsqueeze(0)

        # Generate stft and use torchaudio's `phase_vocoder` to change the tempo
        # Ref: https://pytorch.org/audio/stable/functional.html?highlight=phase_vocoder#torchaudio.functional.phase_vocoder
        phase_advance = torch.linspace(
            0, math.pi * self.stft_hop_length, self.num_freq, device=input_tensor.device
        )[..., None]
        window = torch.hann_window(self.stft_win_length, device=input_tensor.device)

        batch_size = input_tensor.shape[0]
        outputs = []
        output_lengths = []

        for i in range(batch_size):
            if self.stretch_factor is None:
                stretch_factor = Uniform(*self.stretch_factor_range).sample().item()
            else:
                stretch_factor = self.stretch_factor
            stft_output = torch.stft(
                input=input_tensor[i, : int(input_lengths[i].item())],
                n_fft=self.stft_nfft,
                hop_length=self.stft_hop_length,
                win_length=self.stft_win_length,
                window=window,
                return_complex=True,
            )
            stretched_stft_output = torchaudio.functional.phase_vocoder(
                stft_output,
                stretch_factor,
                phase_advance,
            )
            # Now reconstruct the audio from the stretched spectrum
            stretched_input = torch.istft(
                input=stretched_stft_output,
                n_fft=self.stft_nfft,
                hop_length=self.stft_hop_length,
                win_length=self.stft_win_length,
                window=window,
                return_complex=False,
            )
            outputs.append(stretched_input)
            output_lengths.append(stretched_input.shape[-1])

        padded_outputs = pad_sequence(outputs, batch_first=True)
        output_lengths_tensor = torch.IntTensor(output_lengths).to(input_tensor.device)

        if not batched_input:
            padded_outputs = padded_outputs.squeeze(0)

        return padded_outputs, output_lengths_tensor



class DropSampleAugment(Augmentation):
    """
    Randomly drops few samples from an audio array
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        frame_size: int = 400,
        min_drop: int = 4,
        max_drop: int = 40,
    ):
        """
        Attributes
        """
        super().__init__(False, True)
        self.frame_size = frame_size
        self.min_drop = min_drop
        self.max_drop = max_drop

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        drop = random.randint(self.min_drop, self.max_drop)
        drop_axis = input_tensor.shape[-1]

        for i in range(0, drop_axis, self.frame_size):
            input_tensor[:, i : i + self.frame_size][:, -drop:] = 0

        return input_tensor, input_lengths

    @classmethod
    def from_cfg(cls: Type["DropSampleAugment"], cfg: DictConfig) -> "DropSampleAugment":
        return cls(
            cfg.frame_size,
            cfg.min_drop,
            cfg.max_drop,
        )


class LowShelfFilter(Augmentation):
    """
    A low shelf filter is a filter that either boosts (increases amplitude) or cuts (decreases
    amplitude) frequencies below a certain center frequency. This transform applies a low-shelf
    filter at a specific center frequency in hertz. The gain at DC frequency is controlled by
    `{min,max}_gain_db` (note: can be positive or negative!). Filter coefficients are taken from the
    W3 Audio EQ Cookbook: https://www.w3.org/TR/audio-eq-cookbook/

    Ported from audiomentations library: https://github.com/iver56/audiomentations/blob/v0.38.0/audiomentations/augmentations/low_shelf_filter.py

    NOTE: Results may vary slightly compared to the audiomentations implementation due to the use of
    Torch's lfilter instead of SciPy's sosfilt method
    """

    def __init__(
        self,
        sample_rate: int,
        prob: float,
        min_center_frequency: float = 50.0,
        max_center_frequency: float = 4000.0,
        min_gain_db: float = -18.0,
        max_gain_db: float = 18.0,
        min_q: float = 0.1,
        max_q: float = 0.999,
    ) -> None:
        """
        Parameters
        ----------
        sample_rate: ``int``
            Sample rate of the input audio
        prob: ``float``
            Probability of applying the augmentation
        min_center_frequency: ``float`` ( default = 50.0 )
            Minimum center frequency of the low shelf filter
        max_center_frequency: ``float`` ( default = 4000.0 )
            Maximum center frequency of the low shelf filter
        min_gain_db: ``float`` ( default = -18.0 )
            Minimum gain in dB
        max_gain_db: ``float`` ( default = 18.0 )
            Maximum gain in dB
        min_q: ``float`` ( default = 0.1 )
            Min Q factor of the low shelf filter The higher the Q, the narrower the filter
        max_q: ``float`` ( default = 0.999 )
            Max Q factor of the low shelf filter The higher the Q, the narrower the filter
        """
        super().__init__(False, True)

        self.sample_rate = sample_rate
        self.prob = prob
        self.min_center_frequency = min_center_frequency
        self.max_center_frequency = max_center_frequency
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.min_q = min_q
        self.max_q = max_q

    @staticmethod
    def _get_biquad_coefficients(
        sample_rate: int, center_freq: float, gain_db: float, q_factor: float
    ) -> tuple[float, float, float, float, float, float]:
        normalized_frequency = 2 * torch.pi * torch.tensor(center_freq) / sample_rate
        gain = torch.tensor(10 ** (gain_db / 40))
        alpha = torch.sin(normalized_frequency) / 2 / q_factor

        b0 = gain * ((gain + 1) - (gain - 1) * torch.cos(normalized_frequency) + 2 * torch.sqrt(gain) * alpha)

        b1 = 2 * gain * ((gain - 1) - (gain + 1) * torch.cos(normalized_frequency))

        b2 = gain * ((gain + 1) - (gain - 1) * torch.cos(normalized_frequency) - 2 * torch.sqrt(gain) * alpha)

        a0 = (gain + 1) + (gain - 1) * torch.cos(normalized_frequency) + 2 * torch.sqrt(gain) * alpha

        a1 = -2 * ((gain - 1) + (gain + 1) * torch.cos(normalized_frequency))

        a2 = (gain + 1) + (gain - 1) * torch.cos(normalized_frequency) - 2 * torch.sqrt(gain) * alpha

        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        a0 /= a0

        return b0.item(), b1.item(), b2.item(), a0.item(), a1.item(), a2.item()

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.prob:
            return input_tensor, input_lengths

        center_frequency = clamp_frequency(
            random.uniform(self.min_center_frequency, self.max_center_frequency), self.sample_rate
        )
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        q_factor = random.uniform(self.min_q, self.max_q)

        b0, b1, b2, a0, a1, a2 = self._get_biquad_coefficients(self.sample_rate, center_frequency, gain_db, q_factor)

        # Torch recommends using float64 to minimize numerical errors
        # Ref: https://pytorch.org/audio/2.0.1/generated/torchaudio.functional.lfilter.html
        input_dtype = input_tensor.dtype
        input_tensor = torchaudio.functional.biquad(
            waveform=input_tensor.to(torch.float64), b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2
        )

        return input_tensor.to(input_dtype), input_lengths


class HighShelfFilter(Augmentation):
    """
    A high shelf filter is a filter that either boosts (increases amplitude) or cuts(decreases
    amplitude) frequencies above a certain center frequency. This transform applies a high-shelf
    filter at a specific center frequency in hertz. The gain at Nyquist frequency is controlled by
    `{min,max}_gain_db` (note: can be positive or negative!). Filter coefficients are taken from the
    W3 Audio EQ Cookbook: https://www.w3.org/TR/audio-eq-cookbook/

    Ported from audiomentations library: https://github.com/iver56/audiomentations/blob/v0.38.0/audiomentations/augmentations/high_shelf_filter.py

    NOTE: Results may vary slightly compared to the audiomentations implementation due to the use of
    Torch's lfilter instead of SciPy's sosfilt method
    """

    def __init__(
        self,
        sample_rate: int,
        prob: float,
        min_center_frequency: float = 300.0,
        max_center_frequency: float = 7500.0,
        min_gain_db: float = -18.0,
        max_gain_db: float = 18.0,
        min_q: float = 0.1,
        max_q: float = 0.999,
    ) -> None:
        """
        Parameters
        ----------
        sample_rate: ``int``
            Sample rate of the input audio
        prob: ``float``
            Probability of applying the augmentation
        min_center_frequency: ``float`` ( default = 300.0 )
            Minimum center frequency of the high shelf filter
        max_center_frequency: ``float`` ( default = 7500.0 )
            Maximum center frequency of the high shelf filter
        min_gain_db: ``float`` ( default = -18.0 )
            Minimum gain in dB
        max_gain_db: ``float`` ( default = 18.0 )
            Maximum gain in dB
        min_q: ``float`` ( default = 0.1 )
            Minimum quality factor of the high shelf filter
        max_q: ``float`` ( default = 0.999 )
            Maximum quality factor of the high shelf filter
        """
        super().__init__(False, True)

        self.sample_rate = sample_rate
        self.prob = prob
        self.min_center_frequency = min_center_frequency
        self.max_center_frequency = max_center_frequency
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.min_q = min_q
        self.max_q = max_q

    @staticmethod
    def _get_biquad_coefficients(
        sample_rate: int, center_freq: float, gain_db: float, q_factor: float
    ) -> tuple[float, float, float, float, float, float]:
        normalized_frequency = 2 * torch.pi * torch.tensor(center_freq) / sample_rate
        gain = torch.tensor(10 ** (gain_db / 40))
        alpha = torch.sin(normalized_frequency) / 2 / q_factor

        b0 = gain * ((gain + 1) + (gain - 1) * torch.cos(normalized_frequency) + 2 * torch.sqrt(gain) * alpha)

        b1 = -2 * gain * ((gain - 1) + (gain + 1) * torch.cos(normalized_frequency))

        b2 = gain * ((gain + 1) + (gain - 1) * torch.cos(normalized_frequency) - 2 * torch.sqrt(gain) * alpha)

        a0 = (gain + 1) - (gain - 1) * torch.cos(normalized_frequency) + 2 * torch.sqrt(gain) * alpha

        a1 = 2 * ((gain - 1) - (gain + 1) * torch.cos(normalized_frequency))

        a2 = (gain + 1) - (gain - 1) * torch.cos(normalized_frequency) - 2 * torch.sqrt(gain) * alpha

        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        a0 /= a0

        return b0.item(), b1.item(), b2.item(), a0.item(), a1.item(), a2.item()

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.prob:
            return input_tensor, input_lengths

        center_frequency = clamp_frequency(
            random.uniform(self.min_center_frequency, self.max_center_frequency), self.sample_rate
        )
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        q_factor = random.uniform(self.min_q, self.max_q)

        b0, b1, b2, a0, a1, a2 = self._get_biquad_coefficients(self.sample_rate, center_frequency, gain_db, q_factor)

        # Torch recommends using float64 to minimize numerical errors
        # Ref: https://pytorch.org/audio/2.0.1/generated/torchaudio.functional.lfilter.html
        input_dtype = input_tensor.dtype
        input_tensor = torchaudio.functional.biquad(
            waveform=input_tensor.to(torch.float64), b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2
        )

        return input_tensor.to(input_dtype), input_lengths


class Aliasing(Augmentation):
    """
    Apply aliasing to the input audio signal by resampling it to a lower sample rate without
    filtering and then resampling it back to the original sample rate

    Ported from audiomentations library: https://github.com/iver56/audiomentations/blob/v0.38.0/audiomentations/augmentations/aliasing.py
    """

    def __init__(
        self,
        prob: float,
        sample_rate: int,
        min_sample_rate: int = 4000,
        max_sample_rate: int = 8000,
    ) -> None:
        """
        Parameters
        ----------
        prob: ``float``
            Probability of applying the augmentation
        sample_rate: ``int``
            Sample rate of the input audio
        min_sample_rate: ``int`` ( default = 4000 )
            Minimum sample rate to resample the audio
        max_sample_rate: ``int`` ( default = 8000 )
            Maximum sample rate to resample the audio

        Raises
        ------
        ``ValueError``
            - If min_sample_rate is less than 2
            - If min_sample_rate is greater than max_sample_rate
            - If min_sample_rate or max_sample_rate is greater than or equal to the input sample
            rate
        """
        super().__init__(False, True)

        if min_sample_rate < 2:
            raise ValueError("min_sample_rate must be greater than or equal to 2")

        if min_sample_rate > max_sample_rate:
            raise ValueError("min_sample_rate must not be larger than max_sample_rate")

        if min_sample_rate >= sample_rate or max_sample_rate >= sample_rate:
            raise ValueError("min_sample_rate and max_sample_rate must be less than than the input sample rate")

        self.prob = prob
        self.sample_rate = sample_rate
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.prob:
            return input_tensor, input_lengths

        device = input_tensor.device
        input_dtype = input_tensor.dtype

        n = input_tensor.shape[-1]
        new_sample_rate = torch.randint(self.min_sample_rate, self.max_sample_rate + 1, (1,)).item()

        x = torch.linspace(0, n, steps=n, dtype=torch.float64, device=device).expand(input_tensor.shape[0], -1)
        dwn_n = round(n * float(new_sample_rate) / self.sample_rate)
        dwn_x = torch.linspace(0, n, steps=dwn_n, dtype=torch.float64, device=device).expand(input_tensor.shape[0], -1)

        distorted_tensor = interp(x, dwn_x, interp(dwn_x, x, input_tensor))

        return distorted_tensor.to(input_dtype), input_lengths


class AddGaussianNoise(Augmentation):
    """
    Add Gaussian noise to the input audio signal with a given probability

    Ported from the audiomentations library: https://github.com/iver56/audiomentations/blob/v0.38.0/audiomentations/augmentations/add_gaussian_noise.py
    """

    def __init__(
        self,
        prob: float,
        min_amplitude: float = 0.001,
        max_amplitude: float = 0.015,
        batch_agnostic: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        prob: ``float``
            Probability of applying the augmentation
        min_amplitude: ``float``
            Minimum amplitude of the Gaussian noise
        max_amplitude: ``float``
            Maximum amplitude of the Gaussian noise
        batch_agnostic: ``bool`` ( default = False )
            If True, the same noise is added to all samples in the batch. If False, different noise
            is added to each sample in the batch

        Raises
        ------
        ``ValueError``
            If min_amplitude is greater than max_amplitude
        """
        super().__init__(False, batch_agnostic)

        if min_amplitude > max_amplitude:
            raise ValueError("min_amplitude must be less than or equal to max_amplitude")

        self.prob = prob
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.prob:
            return input_tensor, input_lengths

        amplitude = torch.tensor(random.uniform(self.min_amplitude, self.max_amplitude))

        if self.is_batch_agnostic:
            input_tensor = input_tensor + amplitude * torch.randn(
                input_tensor.shape[-1], device=input_tensor.device, dtype=input_tensor.dtype
            ).expand_as(input_tensor)
        else:
            input_tensor = input_tensor + amplitude * torch.randn_like(input_tensor)

        return input_tensor, input_lengths


class NoiseAugment(Augmentation):
    """
    Add noise to the original for Noise Augmentation

    The noise augmentation process is as follows:
        1: Randomly sample noise audios from noise dataset
        2: Extract noise from `audio_paths`
        3: Add noise to original audio
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        noise_dir: str,
        snr: Union[float, List[float]] = [0, 20],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        super().__init__(False, False)
        self.noise_dataset_dir = noise_dir
        self.noise_audio_files = os.listdir(noise_dir)
        self.snr = snr
        self.sample_rate = sample_rate

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_device = input_tensor.device
        input_array = input_tensor.detach().cpu().numpy()

        noise_level: float
        if isinstance(self.snr, float):
            noise_level = self.snr
        elif isinstance(self.snr, int):
            noise_level = (float)(self.snr)
        else:
            noise_level = np.random.uniform(self.snr[0], self.snr[1])

        noise_audio_file = os.path.join(self.noise_dataset_dir, np.random.choice(self.noise_audio_files))
        audio_segment: AudioSegment = AudioSegment.from_file(
            noise_audio_file,
            target_sr=DEFAULT_SAMPLE_RATE,
        )
        
        noise_audio_array = np.array(audio_segment.samples)
        noise_audio_array = noise_audio_array.clip(-1, 1)

        assert isinstance(input_array, np.ndarray)  # make mypy happy
        
        try:
            noise_speech, _, _ = add_background_noise(
                input_array, noise_audio_array, snr_db=noise_level, sample_rate=self.sample_rate
            )
        except Exception as e:
            print(f"min max input: {input_array.min()}, {input_array.max()}")
            print(f"min max noise: {noise_audio_array.min()}, {noise_audio_array.max()}")
            print(f"noise file: {noise_audio_file}")
            raise e
        noise_speech_tensor = torch.from_numpy(noise_speech).to(input_device)

        # noise_speech would be of length max(clean_length or noise_length)
        # assert noise_speech_tensor.shape == input_array.shape or noise_speech_tensor.shape == noise_audio_array.shape

        return noise_speech_tensor, input_lengths

    @classmethod
    def from_cfg(cls: Type["NoiseAugment"], cfg: DictConfig) -> "NoiseAugment":
        return cls(
            cfg.noise_dataset_dir,
            cfg.snr,
            cfg.sample_rate,
        )


class RevEcho(Augmentation):
    """
    This is taken from Facebook's denoiser repository at
    https://github.com/facebookresearch/denoiser/blob/master/denoiser/augment.py#L29 .

    Below are the changes made to this version:
        * Original audio input is with (batch, channels, length) size, here we cut off the
        channel dimension
        * `forward()` method takes a tuple of clean and mixed inputs in the original version.
        Here we only consider one audio input at a time
        * Originally applies the transformation with a given proabability. It is not done here.

    Hacky Reverb but runs on GPU without slowing down training.
    This reverb adds a succession of attenuated echos of the input
    signal to itself. Intuitively, the delay of the first echo will happen
    after roughly 2x the radius of the room and is controlled by `first_delay`.
    Then RevEcho keeps adding echos with the same delay and further attenuation
    until the amplitude ratio between the last and first echo is 1e-3.
    The attenuation factor and the number of echos to adds is controlled
    by RT60 (measured in seconds). RT60 is the average time to get to -60dB
    (remember volume is measured over the squared amplitude so this matches
    the 1e-3 ratio).
    At each call to RevEcho, `first_delay`, `initial` and `RT60` are
    sampled from their range. Then, to prevent this reverb from being too regular,
    the delay time is resampled uniformly within `first_delay +- 10%`,
    as controlled by the `jitter` parameter. Finally, for a denser reverb,
    multiple trains of echos are added with different jitter noises.
    Args:
        - initial: amplitude of the first echo as a fraction
            of the input signal. For each sample, actually sampled from
            `[0, initial]`. Larger values means louder reverb. Physically,
            this would depend on the absorption of the room walls.
        - rt60: range of values to sample the RT60 in seconds, i.e.
            after RT60 seconds, the echo amplitude is 1e-3 of the first echo.
            The default values follow the recommendations of
            https://arxiv.org/ftp/arxiv/papers/2001/2001.08662.pdf, Section 2.4.
            Physically this would also be related to the absorption of the
            room walls and there is likely a relation between `RT60` and
            `initial`, which we ignore here.
        - first_delay: range of values to sample the first echo delay in seconds.
            The default values are equivalent to sampling a room of 3 to 10 meters.
        - repeat: how many train of echos with differents jitters to add.
            Higher values means a denser reverb.
        - jitter: jitter used to make each repetition of the reverb echo train
            slightly different. For instance a jitter of 0.1 means
            the delay between two echos will be in the range `first_delay +- 10%`,
            with the jittering noise being resampled after each single echo.
        - keep_clean: fraction of the reverb of the clean speech to add back
            to the ground truth. 0 = dereverberation, 1 = no dereverberation.
        - sample_rate: sample rate of the input signals.
    """

    def __init__(
        self,
        prob=0.5,
        initial=0.3,
        rt60=(0.3, 1.3),
        first_delay=(0.01, 0.03),
        repeat=3,
        jitter=0.1,
        keep_clean=0.1,
        sample_rate=DEFAULT_SAMPLE_RATE,
    ):
        super().__init__(False, True)
        self.prob = prob
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.sample_rate = sample_rate

    def _reverb(self, source, initial, first_delay, rt60):
        """
        Return the reverb for a single source.
        """
        length = source.shape[-1]
        reverb = torch.zeros_like(source)
        for _ in range(self.repeat):
            frac = 1  # what fraction of the first echo amplitude is still here
            echo = initial * source
            while frac > 1e-3:
                # First jitter noise for the delay
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                delay = min(1 + int(jitter * first_delay * self.sample_rate), length)
                # Delay the echo in time by padding with zero on the left
                echo = F.pad(echo[:, :-delay], (delay, 0))
                reverb += echo

                # Second jitter noise for the attenuation
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                # we want, with `d` the attenuation, d**(rt60 / first_ms) = 1e-3
                # i.e. log10(d) = -3 * first_ms / rt60, so that
                attenuation = 10 ** (-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation
        return reverb

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.prob:
            return input_tensor, input_lengths

        # Sample characteristics for the reverb
        initial = random.random() * self.initial
        first_delay = random.uniform(*self.first_delay)
        rt60 = random.uniform(*self.rt60)

        reverb = self._reverb(input_tensor, initial, first_delay, rt60)
        # Split clean reverb among the clean speech and noise
        input_tensor = input_tensor + self.keep_clean * reverb

        return input_tensor, input_lengths


class AudioMix(Augmentation):

    """
    Mix two audio signals together
    """
    def __init__(self, language_dir:str|Path, segments_dir:str|Path, lang_probs:list[tuple[str, float]]=None, max_audio_length:float=30, prob:float=0.3,num_other_lang_files:int=3):

        super().__init__(False, False)
        
        if isinstance(language_dir, str):
            language_dir = Path(language_dir)
        if isinstance(segments_dir, str):
            segments_dir = Path(segments_dir)
        if not language_dir.exists() or not language_dir.is_dir():
            raise ValueError(f"Language directory {language_dir} does not exist or is not a directory")
        if not segments_dir.exists() or not segments_dir.is_dir():
            raise ValueError(f"Segments directory {segments_dir} does not exist or is not a directory")
        
        self.prob = prob
        self.language_dir = language_dir
        self.lang_folders= [folder for folder in language_dir.glob("*") if folder.is_dir()]
        language_codes = [folder.name for folder in self.lang_folders]
        
        if lang_probs is None:
            lang_probs = [(lang.name, 1/len(self.lang_folders)) for lang in self.lang_folders]
            
        self.lang_probs = lang_probs
        self.max_audio_length = max_audio_length*DEFAULT_SAMPLE_RATE
        self.min_audio_length = 2*DEFAULT_SAMPLE_RATE
        
        
        for lang, lang_prob in lang_probs:
            if lang not in language_codes:
                raise ValueError(f"Language {lang} not found in {language_dir}")
            if lang_prob < 0 or lang_prob > 1:
                raise ValueError(f"Probability for {lang} should be between 0 and 1")
        
        total_prob = sum(lang_prob for _, lang_prob in lang_probs)
        if not math.isclose(total_prob, 1.0):
            raise ValueError("Total probabilities must sum to 1.")
        
        self.lang_audio_files=[]
        for lang_dir in self.lang_folders:
            if not lang_dir.is_dir():
                print(f"Skipping {lang_dir} as it is not a directory")
                continue
            self.lang_audio_files.append(list(lang_dir.glob("*")))
        
        
        for i in range(len(self.lang_audio_files)):
            if len(self.lang_audio_files[i])==0:
                raise ValueError(f"No audio files found in {self.lang_audio_files[i]}")
            logging.info(f"Language {self.lang_folders[i].name} has {len(self.lang_audio_files[i])} audio files")
            
        self.segments_dir = Path(segments_dir)
        self.num_other_lang_files = num_other_lang_files
    
    def get_aug_idx(self,aug_audio_len,audio_len):
        max_possible_len=self.max_audio_length-audio_len
        
        min_possible_len=self.min_audio_length
        possible_duration=random.randint(min_possible_len,max_possible_len)

        if possible_duration>aug_audio_len:
            possible_duration=aug_audio_len

        start_idx=random.randint(0,aug_audio_len-possible_duration)
        end_idx=min(start_idx+possible_duration,aug_audio_len)
        
        return start_idx,end_idx
    

    def transform(self, input_tensor: torch.Tensor, input_lengths: torch.Tensor, input_file_name, aug_lang_idx=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample the language to augment
        
            
        if aug_lang_idx is None:
            aug_lang_idx = random.choices(range(len(self.lang_probs)), weights=[prob for _, prob in self.lang_probs], k=1)[0]
            
        lang_audio_files = self.lang_audio_files[aug_lang_idx]
        
        # Sample the audio file from the selected language
        audio_file = random.choice(lang_audio_files)
        

        aug_audio_segment: AudioSegment = AudioSegment.from_file(
            audio_file,
            target_sr=DEFAULT_SAMPLE_RATE,
        )
        aug_audio = torch.tensor(aug_audio_segment.samples)
        
        # # Calculate the start and end indices for the augmentation audio
        aug_start_idx,aug_end_idx=self.get_aug_idx(len(aug_audio),int(input_lengths.item()))
        
        aug_audio=scale_to_target_signal(aug_audio, input_tensor)
        
        cut_aug_audio= aug_audio[aug_start_idx:aug_end_idx]
        
        # Sample a segment from the input audio segment to insert the other language audio
        segments_file = self.segments_dir / f"{Path(input_file_name).stem}.json"
        
        with open(segments_file, 'r') as f:
            segment = json.load(f)['segments']
        segment.insert(0, {"start": 0, "end": 0})
        
        cut_segment= random.choice(segment)
        
        # Calculate the start and end indices for the input audio
        start_idx= int(cut_segment['end']* DEFAULT_SAMPLE_RATE)
        # print(f"Adding {len(cut_aug_audio)} samples of {audio_file} to {input_file_name} at {start_idx}")

        transformed_audio = torch.cat((input_tensor[:start_idx], cut_aug_audio, input_tensor[start_idx:]))
        # Ensure the transformed audio is within the valid range
        
        # if random.random() < 0.5:
        #     transformed_audio = torch.cat((input_tensor, cut_aug_audio))
        # else:
        #     transformed_audio = torch.cat((cut_aug_audio, input_tensor))
        

        # return transformed_audio, torch.tensor(len(transformed_audio)), audio_file, start_idx, aug_start_idx, aug_end_idx, cut_aug_audio
        return transformed_audio, torch.tensor(len(transformed_audio))

perturbation_types = {
    "speed_perturbation": SpeedPerturbation,
    "gain": Gain,
    "time_stretch": TimeStretch,
    "drop_sample_augment": DropSampleAugment,
    "low_shelf_filter": LowShelfFilter,
    "high_shelf_filter": HighShelfFilter,
    "aliasing": Aliasing,
    "add_gaussian_noise": AddGaussianNoise,
    "noise_augment": NoiseAugment,
    "reverb_echo": RevEcho,
    "audio_mix": AudioMix,
}

def register_perturbation(name: str, augmentaion: Augmentation):
    if name in perturbation_types.keys():
        raise KeyError(
            f"Perturbation with the name {name} exists. " f"Type of perturbation : {perturbation_types[name]}."
        )

    perturbation_types[name] = augmentaion


class CustomAudioAugmentor(object):
    def __init__(self, augmentations:list[list[Augmentation]]=None, rng=None,prob=0.5):
        random.seed(rng) if rng else None
        self._augmentations = augmentations if augmentations is not None else []
        self.prob = prob
    
    @property
    def augmentations(self):
        return self._augmentations

    def __call__(self, audio_signal,audio_signal_length,aug_idx=None):
        
        if aug_idx is not None:
            
                
            for augment in self._augmentations[aug_idx]:
                if audio_signal.ndim == 1:
                    audio_signal = audio_signal.unsqueeze(0)
                if audio_signal_length.ndim==0:
                    audio_signal_length = audio_signal_length.unsqueeze(0)
                audio_signal, audio_signal_length = augment(audio_signal, audio_signal_length)
                audio_signal=torch.clamp(audio_signal, -1, 1)
                if audio_signal.max()==torch.nan or audio_signal.min()==torch.nan or audio_signal.max()>1 or audio_signal.min()< -1:
                    raise ValueError(
                        f"Transformed audio {audio_signal.max()} {audio_signal.min()} is out of range [-1, 1] while applying augment sequence: {self._augmentations[aug_idx]} and current augment: {augment} values ge 1: {(audio_signal>1).sum()} values le -1: {(audio_signal<-1).sum()} audio signal length: {audio_signal_length}"
                    )
        else:
            logging.warning("No augmentation index provided. Skipping.")
        
        if audio_signal.ndim == 2:
            audio_signal = audio_signal.squeeze(0)
        
        if audio_signal_length.ndim == 1:
            audio_signal_length = audio_signal_length.squeeze(0)
            
        return audio_signal, audio_signal_length

    def max_augmentation_length(self, length):
        return length

    @classmethod
    def from_config(cls, config):
        ptbs = []
        for p in config:
            if p['aug_type'] not in perturbation_types:
                logging.warning("%s perturbation not known. Skipping.", p['aug_type'])
                continue
            perturbation = perturbation_types[p['aug_type']]
            ptbs.append((p['prob'], perturbation(**p['cfg'])))
        return cls(perturbations=ptbs)


def process_custom_augmentations(augmenter, global_rank=0, world_size=1) -> Optional[CustomAudioAugmentor]:
    """Process list of online data augmentations.
    Accepts either an AudioAugmentor object with pre-defined augmentations,
    or a dictionary that points to augmentations that have been defined.
    If a dictionary is passed, must follow the below structure:
    Dict[str, Dict[str, Any]]: Which refers to a dictionary of string
    names for augmentations, defined in `asr/parts/perturb.py`.
    The inner dictionary may contain key-value arguments of the specific
    augmentation, along with an essential key `prob`. `prob` declares the
    probability of the augmentation being applied, and must be a float
    value in the range [0, 1].
    # Example in YAML config file
    Augmentations are generally applied only during training, so we can add
    these augmentations to our yaml config file, and modify the behaviour
    for training and evaluation.
    ```yaml
    AudioToSpeechLabelDataLayer:
        ...  # Parameters shared between train and evaluation time
        train:
            augmentor:
                shift:
                    prob: 0.5
                    min_shift_ms: -5.0
                    max_shift_ms: 5.0
                white_noise:
                    prob: 1.0
                    min_level: -90
                    max_level: -46
                ...
        eval:
            ...
    ```
    Then in the training script,
    ```python
    import copy
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(model_config) as f:
        params = yaml.load(f)
    # Train Config for Data Loader
    train_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    train_dl_params.update(params["AudioToTextDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]
    data_layer_train = nemo_asr.AudioToTextDataLayer(
        ...,
        **train_dl_params,
    )
    # Evaluation Config for Data Loader
    eval_dl_params = copy.deepcopy(params["AudioToTextDataLayer"])
    eval_dl_params.update(params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]
    data_layer_eval = nemo_asr.AudioToTextDataLayer(
        ...,
        **eval_dl_params,
    )
    ```
    # Registering your own Augmentations
    To register custom augmentations to obtain the above convenience of
    the declaring the augmentations in YAML, you can put additional keys in
    `perturbation_types` dictionary as follows.
    ```python
    from nemo.collections.asr.parts import perturb
    # Define your own perturbation here
    class CustomPerturbation(perturb.Perturbation):
        ...
    perturb.register_perturbation(name_of_perturbation, CustomPerturbation)
    ```
    Args:
        augmenter: AudioAugmentor object or
            dictionary of str -> kwargs (dict) which is parsed and used
            to initialize an AudioAugmentor.
            Note: It is crucial that each individual augmentation has
            a keyword `prob`, that defines a float probability in the
            the range [0, 1] of this augmentation being applied.
            If this keyword is not present, then the augmentation is
            disabled and a warning is logged.
    Returns: AudioAugmentor object
    """
    if augmenter is None:
        return None

    if isinstance(augmenter, CustomAudioAugmentor):
        return augmenter

    if HAVE_OMEGACONG_WEBDATASET and isinstance(augmenter, DictConfig):
        augmenter = OmegaConf.to_container(augmenter, resolve=True)

    augmenter = copy.deepcopy(augmenter)
    prob=0.5
    if "prob" in augmenter:
        prob = augmenter["prob"]
        del augmenter["prob"]
    if "pipeline" not in augmenter:
        raise KeyError("pipeline not found in augmenter")

    total_augmentations = []
    audio_mix_augmentor=None
    for augmentations in augmenter['pipeline']:
        curr_augmentations=[]
        for augmentation in augmentations:
            for augment_name, augment_kwargs in augmentation.items():
                try:
                    augmentation_class = perturbation_types[augment_name]
                    if 'global_rank' in inspect.signature(augmentation_class).parameters:
                        augment_kwargs['global_rank'] = global_rank
                    if 'world_size' in inspect.signature(augmentation_class).parameters:
                        augment_kwargs['world_size'] = world_size
                    augment = augmentation_class(**augment_kwargs)
                    if isinstance(augment, AudioMix):
                        audio_mix_augmentor= augment
                        continue
                    curr_augmentations.append(augment)
                except KeyError:
                        raise KeyError(f"Invalid perturbation name. Allowed values : {perturbation_types.keys()}")
                    
        if len(curr_augmentations):
            total_augmentations.append(curr_augmentations)
    
    if audio_mix_augmentor is not None:
        total_augmentations.append([audio_mix_augmentor])

    augmenter = CustomAudioAugmentor(augmentations=total_augmentations,prob=prob)
    return augmenter


class AugmentationDataset(IterableDataset):
    """
    A class that loads tarred audio files and cycles over the files in the dataset.
    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset/AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.
    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].
    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.
    See the WebDataset documentation for more information about accepted data and input formats.
    """

    def __init__(
        self,
        manifest_path: str,
        tar_filepaths: Union[str, List[str]],
        shuffle_n: int = 128,
        rank: int = 0,
        world_size: int = 1,
        shard_strategy: str = "replicate",
    ):
        # import here to avoid circular import error
        from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths

        self._manifest = collections.ASRAudioText(manifest_path, parser=parsers.make_parser([]), index_by_file_id=True)

        tar_filepaths = expand_sharded_filepaths(
            tar_filepaths, shard_strategy=shard_strategy, world_size=world_size, global_rank=rank
        )

        if not HAVE_OMEGACONG_WEBDATASET:
            raise LightningNotInstalledException(self)
        self.audio_dataset = wds.DataPipeline(
            wds.SimpleShardList(urls=tar_filepaths),
            wds.shuffle(shuffle_n),
            wds.tarfile_to_samples(),
            wds.rename(audio='wav;ogg;flac', key='__key__'),
            wds.to_tuple('audio', 'key'),
            self._loop_offsets,
        )

    def __len__(self):
        return len(self._manifest)

    def _loop_offsets(self, iterator):
        """This function is used to iterate through utterances with different offsets for each file."""

        class TarredAudioLoopOffsets:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection
                self.current_fn = None
                self.current_bytes = None
                self.offset_id = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.current_fn is None:
                    self.current_bytes, self.current_fn = next(self.iterator)
                    self.offset_id = 0
                else:
                    offset_list = self.collection.mapping[self.current_fn]
                    if len(offset_list) == self.offset_id + 1:
                        self.current_bytes, self.current_fn = next(self.iterator)
                        self.offset_id = 0
                    else:
                        self.offset_id += 1

                return self.current_bytes, self.current_fn, self.offset_id

        return TarredAudioLoopOffsets(self._manifest)

    def __iter__(self):
        audio_iter = iter(self.audio_dataset)

        while True:
            try:
                audio_bytes, audio_filename, offset_id = next(audio_iter)
                file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                manifest_idx = self._manifest.mapping[file_id][offset_id]
                manifest_entry = self._manifest[manifest_idx]

                # Convert audio bytes to IO stream for processing (for SoundFile to read)
                audio_file = io.BytesIO(audio_bytes)
                yield audio_file, file_id, manifest_entry
            except StopIteration:
                audio_iter = iter(self.audio_dataset)
