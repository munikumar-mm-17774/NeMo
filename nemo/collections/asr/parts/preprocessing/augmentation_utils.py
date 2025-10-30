import math
import random

from time import time
from typing import List, Optional, Tuple, Union


import numpy as np
import pyroomacoustics as pra
import torch

from pyroomacoustics import Beamformer, Room

from torch import Tensor

EPS = torch.finfo(torch.float32).eps
DEFAULT_SAMPLE_RATE=16000


def distance_between_coordinates(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.sum((x[:, np.newaxis] - y[:, np.newaxis]) ** 2, axis=0))[0]


def scale_to_target_signal(
    signal_to_scale: np.ndarray,
    target_signal: np.ndarray,
) -> np.ndarray:
    """
    Scales `signal_to_scale` to the range of `target_signal`, so that min and max value
    of both the signals match.

    For example,
    `signal_to_scale`: [0.9378, 0.9171, 0.7226, 0.0634]
    `target_signal`:   [0.1750, 0.0666, 0.0779, 0.0375]
    `scaled_signal`:   [0.1750, 0.1718, 0.1412, 0.0375] (output)
    """
    source_min, source_max = signal_to_scale.min(), signal_to_scale.max()
    target_min, target_max = target_signal.min(), target_signal.max()
    scaled_signal = ((signal_to_scale - source_min) / (source_max - source_min)) * (
        target_max - target_min
    ) + target_min

    return scaled_signal


def power(audio: torch.Tensor) -> torch.Tensor:
    """
    Returns the power of the audio
    """

    return torch.mean(audio**2, dim=-1)


def is_normalized(audio: Union[np.ndarray, torch.Tensor]) -> bool:
    if isinstance(audio, torch.Tensor):
        if (audio.min().item() >= -1.0 and audio.min().item() <= 1.0) and (
            audio.max().item() >= -1.0 and audio.max().item() <= 1.0
        ):
            return True
    else:
        if (audio.min() >= -1.0 and audio.min() <= 1.0) and (audio.max() >= -1.0 and audio.max() <= 1.0):
            return True
    return False


def calculate_padding_length(input_length: int, window_length: int, stride: int) -> int:
    """
    Padding length required to add to input length to ensure that there are no incomplete windows
    when windowing the signal.
    """
    num_windows = math.ceil((input_length - window_length) / stride)
    padding_length = max(0, (num_windows * stride) + window_length - input_length)

    return padding_length


def rms(audio: torch.Tensor) -> torch.Tensor:
    """
    Returns the RMS of the audio
    """

    return torch.sqrt(power(audio))


def amplitude_to_energy_db(amplitude: torch.Tensor) -> torch.Tensor:
    """
    Converts the amplitude to energy in dB
    """

    return 20 * torch.log10(amplitude)


def energy_db(audio: torch.Tensor) -> torch.Tensor:
    """
    Returns the energy of the audio in dB
    """

    return amplitude_to_energy_db(rms(audio) + EPS)


def energy_db_to_power(energy: torch.Tensor) -> torch.Tensor:
    """
    Converts the energy in dB to power
    """

    return 10 ** (energy / 10)


def active_rms(
    clean: Tensor,
    noise: Tensor,
    sample_rate: int,
    energy_threshold_db: float = -50,
    window_size_ms: int = 100,
    hop_size_ms: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Returns the clean and noise RMS of the noise calculated only in the active portions of the noise

    Source: https://github.com/microsoft/DNS-Challenge/blob/master/audiolib.py#L193

    Parameters
    ----------
    clean : ``np.ndarray``
        Input clean audio
    noise : ``np.ndarray``
        Input noise audio
    sample_rate : ``int``
        Audio signal's sample_rate
    energy_threshold_db : ``int`` ( default=-50 )
        Frames with energy below this threshold are not considered for calculating the RMS.
    window_size_ms : ``int`` ( default=100 )
        Window size in milliseconds. For each window, the RMS is calculated and the average of all
        the RMS values is returned.
    hop_size_ms : ``Optional[int]`` ( default = None )
        Hop size in milliseconds. If not provided, it is set to 1/4th of window_size_ms

    Returns
    -------
    ``Tuple[Tensor, Tensor]``
        RMS of clean and noise

    Exceptions
    ----------
    ``ValueError``
        - If clean and noise are not 1D
        - If clean and noise are not of same shape
        - If window_size_ms is less than hop_size_ms
        - If clean and noise are not normalized
    """
    if clean.ndim != 1 or noise.ndim != 1:
        raise ValueError("clean and noise should be 1D")

    if clean.shape != noise.shape:
        raise ValueError("clean and noise should be of same shape")

    hop_size_ms = window_size_ms // 4 if hop_size_ms is None else hop_size_ms
    if window_size_ms < hop_size_ms:
        raise ValueError("window_size_ms should be greater than or equal to hop_size_ms")

    if not (is_normalized(clean) and is_normalized(noise)):
        raise ValueError("clean and noise should be normalized")

    window_length = int(sample_rate * window_size_ms / 1000)
    hop_length = int(sample_rate * hop_size_ms / 1000)

    padding_length = calculate_padding_length(clean.shape[-1], window_length, hop_length)
    clean = torch.cat([clean, torch.zeros(padding_length)])
    noise = torch.cat([noise, torch.zeros(padding_length)])

    clean_windowed = clean.unfold(0, window_length, hop_length)
    noise_windowed = noise.unfold(0, window_length, hop_length)

    noise_energy = energy_db(noise_windowed)
    noise_energy_mask = noise_energy > energy_threshold_db

    if noise_energy_mask.count_nonzero() != 0:
        clean_rms = rms(clean_windowed[noise_energy_mask]).mean()
        noise_rms = rms(noise_windowed[noise_energy_mask]).mean()
    else:
        clean_rms = rms(clean)
        noise_rms = rms(noise)

    return clean_rms, noise_rms


def clamp_frequency(frequency: float, sample_rate: int) -> float:
    """
    Clamps the frequency to the Nyquist frequency of the sample rate

    Parameters
    ----------
    frequency: ``float``
        Input frequency
    sample_rate: ``int``
        Input sample rate

    Returns
    -------
    ``float``
        Clamped frequency
    """
    nyquist_frequency = sample_rate // 2
    if frequency > nyquist_frequency:
        frequency = nyquist_frequency * 0.9999

    return frequency


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    One-dimensional linear interpolation between monotonically increasing sample points, with
    extrapolation beyond sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points (xp, fp), evaluated at x


    Parameters
    ----------
    x: ``torch.Tensor``
        X-coordinates at which to evaluate the interpolated values
    xp: ``torch.Tensor``
        X-coordinates of the data points, must be increasing
    fp: ``torch.Tensor``
        Y-coordinates of the data points, same length as `xp`
    dim: ``int`` ( default = -1 )
        Dimension across which to interpolate

    Returns
    -------
    ``torch.Tensor``
        Interpolated values

    Raises
    ------
    ``ValueError``
        If `xp` and `fp` do not have the same shape
    """
    if xp.shape != fp.shape:
        raise ValueError("xp and fp must have the same shape")

    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
    b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    return values.movedim(-1, dim)


def find_coord_within_distance(reference_point: np.ndarray, distance: float, min_distance: float = 0):
    """
    Function to find another coordinate within a given `distance` from a given `reference_point`
    in 3D space.

    Parameters
    ----------
        reference_point: ``np.ndarray``
            The coordinates of the reference point (x, y, z) of shape (3)
        distance: ``float``
            The maximum distance from the reference point to the generated coordinate.
        min_distance: ``float`` ( default = 0 )
            The minimum distance from the reference point to the generated coordinate.
            If specified, the generated coordinate will be at least this distance away.

    Returns
    -------
        The generated coordinate (x, y, z) within the specified distance range.
    """
    assert reference_point.shape == (3,)
    ax, ay, az = reference_point

    assert distance >= 0

    if distance == 0:
        return ax, ay, az

    # Generate random spherical coordinates
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi)
    r = random.uniform(min_distance, distance)

    # Convert spherical coordinates to Cartesian coordinates
    x = ax + r * math.sin(phi) * math.cos(theta)
    y = ay + r * math.sin(phi) * math.sin(theta)
    z = az + r * math.cos(phi)

    return x, y, z


def add_silence_to_audio_signal(
    audio_signal: Union[np.ndarray, torch.Tensor],
    sample_rate: Optional[int] = None,
    seconds: Optional[Union[int, float]] = None,
    samples: Optional[int] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    This method adds silence to the audio signal at the end of audio.

    Parameters
    ----------
    audio_signal : ``np.ndarray``
        Audio signal in which silence have to be added at the end.
    sample_rate : ``Optional[int]``
        Sample rate of the audio signal.
    seconds : ``Optional[Union[int, float]]``
        Duration of seconds for which silence to be added.
    samples:  ``Optional[int]``
        No of samples of zeros to be added as silence.
    """
    assert samples is not None or all(
        [sample_rate, seconds]
    ), "Either samples or (sample_rate and seconds) should be given"

    if samples is None:
        assert (
            sample_rate is not None and seconds is not None
        ), "sample_rate and seconds both should be given"  # have assert statement to pass mypy check
        no_of_zeros = int(sample_rate * seconds)
    else:
        no_of_zeros = samples

    if isinstance(audio_signal, np.ndarray):
        audio_signal = np.append(audio_signal, np.zeros(no_of_zeros, dtype=audio_signal.dtype))
    else:
        audio_signal = torch.nn.functional.pad(audio_signal, (0, no_of_zeros))

    return audio_signal


def align_clean_and_noise(
    clean: Tensor, noise: Tensor, pad_clean_signal: bool, pad_noise_signal: bool
) -> Tuple[Tensor, Tensor]:
    """
    Align clean and noise signals to match the length of the longer signal

    Parameters
    ----------
    clean : ``Tensor``
        Input clean audio
    noise : ``Tensor``
        Input noise audio
    pad_clean_signal: ``bool``
        - If pad_clean_signal is set to `True`
            - If length of clean is less than the length of noise, zeros will be added in clean to
            match the length of noise
        - If pad_clean_signal is set to `False`
            - noise will be trimmed to match clean length
    pad_noise_signal: ``bool``
        - If pad_noise_signal is set to `True`
            - If length of noise is less than the length of clean, zeros will be added in clean to
            match the length of clean
        - If pad_noise_signal is set to `False`
            - clean will be trimmed to match noise length

    Returns
    -------
    ``Tuple[Tensor, Tensor]``
        Padded clean and noise

    Exceptions
    ----------
    ``ValueError``
        Raised if clean and noise are not 1D
    """
    if clean.ndim != 1 or noise.ndim != 1:
        raise ValueError("clean and noise should be 1D")

    clean_length = len(clean)
    noise_length = len(noise)

    if clean_length == noise_length:
        return clean, noise

    if clean_length > noise_length:
        if pad_noise_signal:
            noise = add_silence_to_audio_signal(noise, samples=clean_length - noise_length)
        clean = clean[: len(noise)]
    else:
        if pad_clean_signal:
            clean = add_silence_to_audio_signal(clean, samples=noise_length - clean_length)
        noise = noise[: len(clean)]

    assert clean.shape == noise.shape

    return clean, noise


class RoomSimulation:

    GLOBAL_DELAY = pra.constants.get("frac_delay_length") // 2

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        """
        Simulates a room using pyroomacoustics that supports physics like audio reflection,
        loudness manipulation using source placement from the microphone, etc,. Currently
        this supports the following methods:

            1. generate_room_impulse_effect - A single audio source is placed in a fixed distance for every call.
            2. mix_using_room_simulation - Mix a multi channel signal by placing each signal at different distances from the mic.
                The output audio is min-max scaled to produce an audible mixed signal.
            3. add_background_noise - Mix clean and noise signals by placing their sources at a very close distance from the mic.

            Functioning of underlying method of `mix_using_room_simulation` and `add_background_noise`:
                Explanation in 3D:
                    - Here we use the term `shell` to define an imaginary spherical shaped
                    boundary with microphone at as its center.
                    - We can pass a batch of audios, each of which exists inside a random shell
                    (between the inner sphere and outer sphere of the shell)
                Explanation in 1D
                    - In case u don't understand the explanation using the term shell, then consider mic at
                    x = 0 on a line on the x axis. A source maybe placed between 0.9 and 1.0. Another source
                    between 1.8 and 2.0 and so on till 5.0.
                    
        Parameters
        ----------
            sample_rate: ``int`` ( default = "medium" )
                Sample rate of the audio signals that will be used as inputs for the Room simulation.
                Internally, processes uniformly all properties that are defined on octave bands for this sample_rate.
        """
        self.fs = sample_rate
        # maximum loudness level that can be simulated. Increasing this value will make
        # the lower levels less audible. Used only by `mix_using_room_simulation`.
        self._max_level: int = 5

    def _prepare_room_for_room_impulse_effect(
        self,
        room_size: str = "medium",
        room_absorption: str = "medium",
    ) -> None:
        # pylint: disable=attribute-defined-outside-init

        if room_size == "large":
            self.size_coef = 5.0
        elif room_size == "medium":
            self.size_coef = 2.5
        elif room_size == "small":
            self.size_coef = 1.0
        else:
            raise ValueError("The size parameter can only take values ['small', 'medium', 'large']")

        if room_absorption == "high":
            self.absorption = 0.7
        elif room_absorption == "medium":
            self.absorption = 0.3
        elif room_absorption == "low":
            self.absorption = 0.1
        else:
            raise ValueError("The absorption parameter can only take values ['low', 'medium', 'high']")

    def _prepare_room_for_mixing(self, max_distance: Union[int, float]) -> None:
        """Scale the room and disable absorption"""
        # pylint: disable=attribute-defined-outside-init

        self.size_coef = (max_distance * 1.25) ** 2
        self.absorption = 1.0

    def setup_room_with_mic(
        self,
        room_dim: Optional[np.ndarray] = None,
        mic_location: Optional[np.ndarray] = None,
        max_order: int = 1,
    ) -> None:
        """
        The simulator has some internal state that needs to be manually reset after changing the sources.
        The easiest solution is to just re-create the object. This does not have a lot of overhead.
        Issue: https://github.com/LCAV/pyroomacoustics/issues/311#issuecomment-1519385858

        Parameters
        ----------
            room_dim: ``Optional[np.ndarray]``  ( default = None )
                List of coordinates of room corners in a 3D space, must be antiClockwise oriented
            mic_location: ``Optional[np.ndarray]``  ( default = None )
                Mics position coordinates in a 3D space
            max_order: ``int``  ( default = 1 )
                The maximum reflection order in the image source model.
        """
        # pylint: disable=attribute-defined-outside-init

        assert isinstance(self.size_coef, float), "Call appropriate _prepare_room method"
        assert isinstance(self.absorption, float), "Call appropriate _prepare_room method"

        small_room_dims = np.array([[0, 0], [0, 4], [3, 2], [3, 0]]) if room_dim is None else room_dim
        pol = self.size_coef * small_room_dims.T
        self.room = Room.from_corners(pol, fs=self.fs, max_order=max_order, absorption=self.absorption)

        # Create the 3D room by extruding the 2D by a specific height
        self.room.extrude(self.size_coef * 2.5, absorption=self.absorption)

        # Adding the microphone
        # mic_coordinates
        self.R = self.size_coef * (np.array([0.5, 1.2, 0.5]) if mic_location is None else mic_location)
        self.room.add_microphone_array(Beamformer(np.expand_dims(self.R, axis=1), self.room.fs))

    @staticmethod
    def is_clipped(audio, clipping_threshold=0.99):
        return any(abs(audio) > clipping_threshold)

    def generate_room_impulse_effect(
        self,
        clean_signal: np.ndarray,
        room_size: str = "medium",
        room_absorption: str = "medium",
    ) -> np.ndarray:
        """
        Generate a room impulse response (RIR) on a `clean_signal` for a given room_size and room_absorption intensity.
        In this method, only a single source is used and is placed at the center of the room.

        Parameters
        ----------
            clean_signal: ``np.ndarray``
                Clean audios signal to be processed. (n_samples,)
            room_size: ``str`` ( default = "medium" )
                Size of the room. The base dimension of the room is trapezoidal and is fixed.
                With three options of room_size `medium` or `large` or `small`, this gets scaled by a multiplicative factor.
            room_absorption: ``str`` ( default = "medium" )
                Absorption of the room. Based on the choice `medium` or `high` or `low`, the room
                absorption is scaled by a multiplicative factor.

        Returns
        -------
            processed_signal: ``np.ndarray``
                RIR Generated processed signal
        """
        self._prepare_room_for_room_impulse_effect(room_size, room_absorption)
        # The simulator has some internal state that needs to be manually reset after changing the sources.
        self.setup_room_with_mic(max_order=10)

        # Location of the source in the room.
        source_loc = np.array([1.8, 0.4, 1.6])

        # Adding the source
        self.room.add_source(
            self.size_coef * np.array([source_loc[0], source_loc[1], source_loc[2]]),
            signal=clean_signal,
        )

        self.room.simulate()
        processed_signal = self.room.mic_array.signals[0, :]

        if self.is_clipped(processed_signal):
            processed_signal_maxamplevel = max(abs(processed_signal)) / (0.99 - EPS)  # 0.99 is clipping threshold
            processed_signal = processed_signal / processed_signal_maxamplevel

        return processed_signal[..., self.GLOBAL_DELAY :]

    def loudness_to_distance(self, lvl: int, max_level: int) -> int:
        """
        Simulating loudness is done by placing source audios near the following distances
        from the mic location - [1m, 4m, 9m, 16m, 25m].
        https://www.csun.edu/scied/6-instrumentation/inverse_square_sound/index.htm

        Signal is attenuated by 1 / distance(source, mic) when traveling from source to microphone.
        Ref - https://github.com/LCAV/pyroomacoustics/issues/261#issuecomment-1147443792
        """
        lvl = max_level - lvl + 1
        return lvl**2

    def distance_to_samples(self, d: float) -> int:
        """
        Considers an ideal atmospheric condition, where the speed of sound is `343 m/s`.
        - time taken for sound to travel distance(`d`): `time_duration = d / 343`
        - number of samples in that time duration: `d / 343 * sample_rate`
        """
        return int(abs(d) / 343 * self.room.fs)

    def _generate_random_source_coordinate(self, distance: float) -> Tuple[float, float, float]:
        """
        Returns random 3D coordinate within `0.9*distance` and `distance` from mic location.
        Has a 2 second timeout (in case all random coordinates are present outside of the room).

        NOTE: In rare cases, the below check `self.room.is_inside` passes but the same check
        present inside `self.room.add_source` fails. The check `self.room.is_inside` is inconsistent.
        References:
            1. https://github.com/LCAV/pyroomacoustics/issues/181
            2. https://github.com/LCAV/pyroomacoustics/issues/248
        """
        timeout = time() + 2
        while time() < timeout:
            # Generate random coordinates within the boundaries
            source_coordinate = find_coord_within_distance(self.R, distance, 0.9 * distance)

            # Check if the generated point is inside the polyhedron
            if self.room.is_inside(source_coordinate, include_borders=False):
                return source_coordinate
        raise ValueError("Source coordinate not found inside the intersection of shell and room")

    def _mix_using_room_simulation(
        self, audio_signal: np.ndarray, source_distances: List[int], align_sources: bool = True
    ) -> np.ndarray:
        """
        Method for mixing audio signals using room simulation by placing sources at different
        distances from the mic.

        Parameters
        ----------
            audio_signal: ``np.ndarray``
                Multi channel signal with audio sources on each channel.
            source_distances: ``List[int]``
                List of maximum distances at which the source signals have to be placed (`0.9*distance` and `distance` from mic location).
                Length of list must match number of input channels.
            align_sources: ``bool`` ( default = True )
                The silence induced due to distance delay for each loudness level will be corrected.

        Returns
        -------
            mixed_signal: ``np.ndarray``
                Signal mixed with room simulation (single channel).
        """
        assert audio_signal.shape[0] == len(source_distances)
        input_dtype = audio_signal.dtype

        # check docstrings of `generate_random_source_coordinate` to understand the need for retrying source placement
        timeout = time() + 2
        while time() < timeout:
            try:
                self._prepare_room_for_mixing(max(source_distances))

                # Here max_order should ideally be set to 0, which keeps only the direct sound from the source.
                # But we want to disable the reflection effect by instead setting wall absorption to 1.
                self.setup_room_with_mic(mic_location=np.array([0.5, 3.2, 0.5]), max_order=1)

                # place sources at randomly generated source coordinates across shells at different distances from the microphone
                # NOTE: distance is not scaled considering how it introduces delay in sound reaching from source to mic
                source_coordinates = []
                for distance in source_distances:
                    source_loc = self._generate_random_source_coordinate(distance)
                    distance_from_mic = distance_between_coordinates(self.R, np.array(source_loc))
                    source_coordinates.append((source_loc, distance_from_mic))
                distance_from_mic_list = [x[1] for x in source_coordinates]

                for i, each_signal in enumerate(audio_signal):
                    assert each_signal.ndim == 1
                    source_loc, distance_from_mic = source_coordinates[i]

                    if align_sources:
                        # all sources are padded with zeros so that all of their actual signals reach the microphone at the same time
                        req_samples = self.distance_to_samples(distance_from_mic - max(distance_from_mic_list))
                        each_signal = np.pad(each_signal, (req_samples, 0))
                    self.room.add_source(np.array([source_loc[0], source_loc[1], source_loc[2]]), signal=each_signal)
                break
            except ValueError as e:
                if str(e) != "The source must be added inside the room.":
                    raise

                continue
        else:
            raise ValueError("Room simulation failed to place sources at random coordinates")

        self.room.simulate()
        # self.room.plot_rir()
        processed_signal = self.room.mic_array.signals[0, :]

        if self.is_clipped(processed_signal):
            processed_signal_maxamplevel = max(abs(processed_signal)) / (0.99 - EPS)  # 0.99 is clipping threshold
            processed_signal = processed_signal / processed_signal_maxamplevel

        # trim signal
        if align_sources:
            # all source signal beginnings were delayed to match with that of the farthest source
            induced_silence_in_samples = self.GLOBAL_DELAY + self.distance_to_samples(max(distance_from_mic_list))
            signal_end = induced_silence_in_samples + audio_signal.shape[-1]
        else:
            # closest source distance is considered for delay adjustment
            induced_silence_in_samples = self.GLOBAL_DELAY + self.distance_to_samples(min(distance_from_mic_list))
            signal_end = (
                self.GLOBAL_DELAY + self.distance_to_samples(max(distance_from_mic_list)) + audio_signal.shape[-1]
            )

        # convert to input dtype
        processed_signal = processed_signal.astype(input_dtype)

        return processed_signal[..., induced_silence_in_samples:signal_end]

    def mix_using_room_simulation(
        self,
        audio_signal: np.ndarray,
        align_sources: bool = True,
        loudness_levels: Optional[List[int]] = None,
        match_input_amplitude: bool = False,
    ) -> np.ndarray:
        """
        Parameters
        ----------
            audio_signal: ``np.ndarray``
                Multi channel signal with audio sources on each channel.
            align_sources: ``bool`` ( default = True )
                The silence induced due to distance delay for each loudness level will be corrected.
            loudness_levels: ``Optional[List[int]]`` ( default = None )
                Performs mixing of audio sources with varied loudness levels. Takes values 1 till 5 (loudest).
                Length of list must match number of input channels.
            match_input_amplitude: ``bool`` ( default = False )
                Simulated output can become inaudible when source is placed very far away from the mic
                (to produced the loudness drop effect). Scaling can be used in such case.

        Returns
        -------
            mixed_signal: ``np.ndarray``
                Signal mixed with room simulation (single channel).
        """
        num_channels: int = len(audio_signal)
        if loudness_levels is not None:
            assert len(loudness_levels) == num_channels
        else:
            loudness_levels = random.choices([1, 2, 3, 4, 5], k=num_channels)
        assert 1 <= max(loudness_levels) <= self._max_level

        # convert loudness levels to distances
        source_distances = [self.loudness_to_distance(lvl, max_level=self._max_level) for lvl in loudness_levels]

        mixed_signal = self._mix_using_room_simulation(
            audio_signal=audio_signal,
            source_distances=source_distances,
            align_sources=align_sources,
        )

        if match_input_amplitude:
            return scale_to_target_signal(mixed_signal, audio_signal.sum(axis=0))
        return mixed_signal

    def add_background_noise(self, clean: np.ndarray, noise: np.ndarray, distance: int = 1) -> np.ndarray:
        """
        Parameters
        ----------
            clean: ``np.ndarray``
                Clean signal.
            noise: ``np.ndarray``
                Scaled noise signal.
            distance: ``int`` ( default = 1 )
                Max distance at which both clean and noise sources will be placed (minimum would be 0.9*distance).
                Signal is attenuated by 1 / distance(source, mic) when traveling from source to microphone.
                Hence, default value of 1m is used.

        Returns
        -------
            mixed_signal: ``np.ndarray``
                Signal mixed with room simulation (single channel).
        """
        return self._mix_using_room_simulation(
            audio_signal=np.stack((clean, noise), axis=0),
            source_distances=[distance, distance],
            align_sources=True,
        )


class AddBackgroundNoise:
    """
    Add noise to clean speech, with specified SNR value to generate mixed speech.

    Attributes
    ----------
    sample_rate : ``int``
        Audio signal's sample_rate.
    pad_clean_signal: ``bool`` `(default=True)`
        - If pad_clean_signal is set to `True`
            - If length of clean is less than the length of noise, zeros will be added in clean to
            match the length of noise
        - If pad_clean_signal is set to `False`
            - noise will be trimmed to match clean length
    pad_noise_signal: ``bool`` `(default=True)`
        - If pad_noise_signal is set to `True`
            - If length of noise is less than the length of clean, zeros will be added in clean to
            match the length of clean
        - If pad_noise_signal is set to `False`
            - clean will be trimmed to match noise length
    use_simulated_mixing: ``bool`` `(default=False)`
        If set to `True`, clean and noise will be mixed using room simulation
    use_active_rms: ``bool`` `(default=False)`
        If set to `True`, Instead of using the entire noise signal for RMS calculation, only the
        active portions of the noise signal will be used.
    active_rms_kwargs: ``Optional[dict]`` `(default=None)`
        Keyword arguments to be passed to ``active_rms`` function.
        - energy_threshold: ``float`` `(default=-50)`
            Frames with energy below this threshold are not considered for calculating the RMS.
        - window_size_ms: ``int`` `(default=100)`
            Window size in milliseconds.
        - hop_size_ms : ``Optional[None]`` ( default = None )
            Hop size in milliseconds. If not provided, it is set to 1/4th of window_size_ms
    """

    def __init__(
        self,
        sample_rate: int,
        pad_clean_signal: bool = True,
        pad_noise_signal: bool = True,
        use_simulated_mixing: bool = False,
        use_active_rms: bool = False,
        active_rms_kwargs: Optional[dict] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.pad_clean_signal = pad_clean_signal
        self.pad_noise_signal = pad_noise_signal
        self.use_simulated_mixing = use_simulated_mixing
        self.use_active_rms = use_active_rms
        self.active_rms_kwargs = active_rms_kwargs or {}

        self._room_simulation: Optional[RoomSimulation] = None
        if self.use_simulated_mixing:
            self._room_simulation = RoomSimulation(sample_rate=self.sample_rate)

    def _get_power(self, clean: Tensor, noise: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns the power of clean and noise signals.

        If ``use_active_rms`` is set to `True`, instead of using the entire noise signal for RMS
        calculation, only the active portions of the noise signal will be used.

        Parameters
        ----------
        clean : ``Tensor``
            Input clean audio
        noise : ``Tensor``
            Input noise audio

        Returns
        -------
        ``Tuple[Tensor, Tensor]``
            Power of clean and noise signals
        """
        if self.use_active_rms:
            clean_rms, noise_rms = active_rms(
                clean=clean, noise=noise, sample_rate=self.sample_rate, **self.active_rms_kwargs
            )
            clean_power = clean_rms**2
            noise_power = noise_rms**2
        else:
            clean_power = power(clean)
            noise_power = power(noise)

        return clean_power, noise_power

    def _noise_scaling_factor(self, clean: Tensor, noise: Tensor, snr_db: float) -> Tensor:
        """
        Returns the scaling factor for noise to be mixed with clean audio to generate mixed audio
        at the given SNR value.

        Parameters
        ----------
        clean : ``Tensor``
            Input clean audio
        noise : ``Tensor``
            Input noise audio

        Returns
        -------
        ``Tensor``
            Scaling factor for noise
        """
        clean_power, noise_power = self._get_power(clean, noise)

        snr_power = energy_db_to_power(torch.tensor(snr_db))
        noise_factor = torch.sqrt(clean_power / torch.maximum((noise_power * snr_power), torch.tensor(EPS)))

        return noise_factor

    def _mix(self, clean: Tensor, noise: Tensor) -> Tensor:
        """
        Returns the mixed audio signal of clean and noise.

        If ``use_simulated_mixing`` is set to `True`, clean and noise will be mixed using room
        simulation else clean and noise will be mixed by adding them.

        Parameters
        ----------
        clean : ``Tensor``
            Input clean audio
        noise : ``Tensor``
            Input noise audio

        Returns
        -------
        ``Tensor``
            Mixed audio signal of clean and noise
        """
        if self.use_simulated_mixing:
            assert self._room_simulation is not None, "room_simulation is not initialized"
            mixed_speech = torch.from_numpy(
                self._room_simulation.add_background_noise(clean.cpu().numpy(), noise.cpu().numpy())
            )
            mixed_speech = mixed_speech.to(clean.device)
        else:
            mixed_speech = clean + noise

        return mixed_speech

    def __call__(self, clean: Tensor, noise: Tensor, snr_db: float) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Augment noise to speech, with specified SNR value to generate mixed speech.

        Parameters
        ----------
        clean : ``np.ndarray``
            Input clean audio
        noise : ``np.ndarray``
            Input clean audio
        snr_db : ``float``
            At the given SNR (decibels) value clean and noise will be mixed to generate mixed

        Returns
        -------
        mixed : ``np.ndarray``
            mixed audio signal
        clean : ``np.ndarray``
            modified clean signal
        noise : ``np.ndarray``
            modified noise signal

        Exceptions
        ----------
        ``ValueError``
            This exception is raised for the following conditions:
            - If clean and noise are not on same device
            - If input clean and noise audios are not normalized
        """
        if clean.device != noise.device:
            raise ValueError("clean and noise should be on same device")

        clean = clean.squeeze()
        noise = noise.squeeze()

        if clean.ndim != 1 or noise.ndim != 1:
            raise ValueError("Input clean and noise audios should be 1D")

        if not (is_normalized(clean) and is_normalized(noise)):
            raise ValueError("Input clean and noise audios should be normalized")

        clean, noise = align_clean_and_noise(clean, noise, self.pad_clean_signal, self.pad_noise_signal)

        scaling_factor = self._noise_scaling_factor(clean, noise, snr_db)
        noise_scaled = noise * scaling_factor
        mixed_speech = self._mix(clean, noise_scaled)

        return mixed_speech, clean, noise_scaled


def add_background_noise(
    clean: Union[Tensor, np.ndarray],
    noise: Union[Tensor, np.ndarray],
    sample_rate: int,
    snr_db: float,
    pad_clean_signal: bool = False,
    pad_noise_signal: bool = True,
    use_simulated_mixing: bool = False,
    use_active_rms: bool = False,
    active_rms_kwargs: Optional[dict] = None,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Functional wrapper for ``AddBackgroundNoise``.

    Refer to ``AddBackgroundNoise`` for more details.

    Parameters
    ----------
    clean : ``np.ndarray``
        Input clean audio
    noise : ``np.ndarray``
        Input clean audio
    sample_rate: ``int``
        Audio signal's sample_rate.
    snr_db: ``float``
        Signal to noise ratio (decibels)
    pad_clean_signal: ``bool`` `(default=True)`
        - If pad_clean_signal is set to `True`
            - If length of clean is less than the length of noise, zeros will be added in clean to
            match the length of noise
        - If pad_clean_signal is set to `False`
            - noise will be trimmed to match clean length
    pad_noise_signal: ``bool`` `(default=True)`
        - If pad_noise_signal is set to `True`
            - If length of noise is less than the length of clean, zeros will be added in clean to
            match the length of clean
        - If pad_noise_signal is set to `False`
            - clean will be trimmed to match noise length
    use_simulated_mixing: ``bool`` `(default=False)`
        If set to `True`, clean and noise will be mixed using room simulation
    use_active_rms: ``bool`` `(default=False)`
        If set to `True`, Instead of using the entire noise signal for RMS calculation, only the
        active portions of the noise signal will be used.
    active_rms_kwargs: ``Optional[dict]`` `(default=None)`
        Keyword arguments to be passed to ``active_rms`` function.
        - energy_threshold: ``float`` `(default=-50)`
            Frames with energy below this threshold are not considered for calculating the RMS.
        - window_size_ms: ``int`` `(default=100)`
            Window size in milliseconds.
        - hop_size_ms : ``Optional[None]`` ( default = None )
            Hop size in milliseconds. If not provided, it is set to 1/4th of window_size_ms

    Returns
    -------
    mixed : ``np.ndarray``
        mixed audio signal
    clean : ``np.ndarray``
        modified clean signal
    noise : ``np.ndarray``
        modified noise signal
    """

    if is_numpy_input := isinstance(clean, np.ndarray):
        clean = torch.from_numpy(clean)
        noise = torch.from_numpy(noise)

    _add_background_noise = AddBackgroundNoise(
        sample_rate=sample_rate,
        pad_clean_signal=pad_clean_signal,
        pad_noise_signal=pad_noise_signal,
        use_simulated_mixing=use_simulated_mixing,
        use_active_rms=use_active_rms,
        active_rms_kwargs=active_rms_kwargs,
    )

    mixed, clean, noise = _add_background_noise(clean, noise, snr_db)  # type: ignore

    if is_numpy_input:
        mixed = mixed.numpy()
        clean = clean.numpy()
        noise = noise.numpy()

    return mixed, clean, noise  # type: ignore
