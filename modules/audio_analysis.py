"""In-memory acoustic feature extraction for research demonstrations."""
from io import BytesIO
import math

import librosa
import numpy as np
import soundfile as sf

from .confidence import confidence_report


MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
MIN_AUDIO_SECONDS = 1.0
MAX_AUDIO_SECONDS = 120.0
TARGET_SAMPLE_RATE = 16000


def _emit(progress_callback, stage, state, message):
    if progress_callback is not None:
        progress_callback(stage, state, message)


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int, dict]:
    """Decode and validate common audio formats while preserving channel metadata."""
    if not isinstance(audio_bytes, (bytes, bytearray, memoryview)) or not audio_bytes:
        raise ValueError("The audio input is empty.")
    if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError(
            f"Audio exceeds the {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )

    raw_bytes = bytes(audio_bytes)
    decoded_with = "soundfile"
    try:
        audio, sample_rate = sf.read(
            BytesIO(raw_bytes),
            dtype="float32",
            always_2d=True,
        )
    except Exception:
        try:
            audio, sample_rate = librosa.load(
                BytesIO(raw_bytes),
                sr=None,
                mono=False,
            )
            decoded_with = "librosa"
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[:, np.newaxis]
            elif audio.ndim == 2:
                audio = audio.T
        except Exception as exc:
            raise ValueError(
                "Could not decode this audio file. Supported: WAV, MP3, M4A, OGG, FLAC."
            ) from exc

    audio = np.asarray(audio, dtype=np.float32)
    sample_rate = int(sample_rate)
    if sample_rate <= 0 or audio.ndim != 2 or not audio.shape[1]:
        raise ValueError("Audio has invalid sample-rate or channel information.")
    if not audio.shape[0] or not np.isfinite(audio).all():
        raise ValueError("Audio contains no samples or invalid numeric values.")

    duration = audio.shape[0] / float(sample_rate)
    if duration < MIN_AUDIO_SECONDS:
        raise ValueError("Please provide at least one second of valid audio.")
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio exceeds the {int(MAX_AUDIO_SECONDS)} second duration limit."
        )

    return audio, sample_rate, {
        "decoded_with": decoded_with,
        "source_sample_rate": sample_rate,
        "source_channels": int(audio.shape[1]),
        "source_seconds": round(duration, 2),
        "source_bytes": len(raw_bytes),
    }


def to_standard_wav(
    audio_bytes: bytes,
    target_sr: int = TARGET_SAMPLE_RATE,
    progress_callback=None,
) -> tuple[bytes, dict]:
    """Decode common audio formats and convert them to mono PCM WAV."""
    _emit(progress_callback, "validation", "running", "Checking file size and decoding audio.")
    audio, source_sr, info = decode_audio_bytes(audio_bytes)
    _emit(
        progress_callback,
        "validation",
        "complete",
        (
            f"Readable audio · {info['source_seconds']:.2f} seconds · "
            f"{info['source_sample_rate']} Hz · {info['source_channels']} channel(s)."
        ),
    )

    _emit(progress_callback, "conversion", "running", "Converting to mono WAV.")
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if source_sr != target_sr:
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=target_sr)

    buffer = BytesIO()
    sf.write(buffer, mono, target_sr, format="WAV", subtype="PCM_16")
    conversion_info = {
        **info,
        "target_sample_rate": int(target_sr),
        "target_channels": 1,
        "converted": (
            source_sr != target_sr
            or info["source_channels"] != 1
            or info["decoded_with"] != "soundfile"
        ),
    }
    _emit(
        progress_callback,
        "conversion",
        "complete",
        f"Prepared mono PCM WAV at {target_sr} Hz.",
    )
    return buffer.getvalue(), conversion_info


def analyze_audio_bytes(
    audio_bytes: bytes,
    progress_callback=None,
    source_sample_rate: int | None = None,
) -> dict:
    if not audio_bytes:
        raise ValueError("The audio input is empty.")
    try:
        audio, sample_rate, _ = decode_audio_bytes(audio_bytes)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Could not decode this audio. Please provide a valid WAV file.") from exc

    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    duration = float(len(audio) / sample_rate)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
    _emit(
        progress_callback,
        "preprocessing",
        "complete",
        "Checked finite mono samples; no additional amplitude normalization was applied.",
    )

    reasons = []
    checks_total = 5
    passed = 0
    if duration >= 5:
        passed += 1
    else:
        reasons.append("Recording is shorter than 5 seconds; repeat with a longer sample.")
    if rms >= 0.003:
        passed += 1
    else:
        reasons.append("Audio level is very low; move closer to the microphone.")
    if clipping_ratio <= 0.01:
        passed += 1
    else:
        reasons.append("Possible clipping detected; reduce input volume.")
    quality_sample_rate = (
        source_sample_rate
        if source_sample_rate is not None
        else sample_rate
    )
    if quality_sample_rate >= TARGET_SAMPLE_RATE:
        passed += 1
    else:
        reasons.append(
            "Source sample rate is below 16 kHz; resampling cannot restore missing detail."
        )

    frame_len = max(1, int(sample_rate * 0.025))
    frame_rms = np.array([
        np.sqrt(np.mean(audio[i:i + frame_len] ** 2))
        for i in range(0, len(audio), frame_len)
        if len(audio[i:i + frame_len])
    ])
    silence_ratio = float(np.mean(frame_rms < 0.001))
    if silence_ratio < 0.8:
        passed += 1
    else:
        reasons.append("Most frames are near-silent; check microphone placement and repeat.")

    _emit(
        progress_callback,
        "feature_extraction",
        "running",
        "Calculating the existing descriptive acoustic features.",
    )
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
        spectral_centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        ))
        rms_frames = librosa.feature.rms(y=audio)[0]
        rms_variability = float(np.std(rms_frames))
    except Exception as exc:
        raise ValueError(
            "Audio feature extraction failed. Try a clean mono WAV file."
        ) from exc

    pitch_median = None
    try:
        f0 = librosa.yin(
            audio,
            fmin=65,
            fmax=min(400, sample_rate / 2 - 1),
            sr=sample_rate,
        )
        voiced = f0[np.isfinite(f0) & (f0 >= 65) & (f0 <= 400)]
        if voiced.size:
            pitch_median = float(np.median(voiced))
    except Exception:
        pitch_median = None

    _emit(
        progress_callback,
        "feature_extraction",
        "complete",
        "Descriptive MFCC, level, silence, spectral, and pitch measurements are ready.",
    )
    quality = confidence_report(reasons, passed, checks_total)
    return {
        "duration_seconds": duration,
        "sample_rate": int(sample_rate),
        "channels": 1,
        "peak_amplitude": peak,
        "rms": rms,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "mfcc_means": np.mean(mfcc, axis=1).astype(float).tolist(),
        "mfcc_stds": np.std(mfcc, axis=1).astype(float).tolist(),
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": spectral_centroid,
        "rms_variability": rms_variability,
        "pitch_median_hz": pitch_median,
        "quality": quality,
    }


def audio_visualization_data(
    audio_bytes: bytes,
    max_waveform_points: int = 1600,
    max_spectrogram_frames: int = 240,
) -> dict:
    """Return bounded waveform and spectrogram arrays for ephemeral UI rendering."""
    audio, sample_rate, _ = decode_audio_bytes(audio_bytes)
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    duration = len(mono) / float(sample_rate)

    frame_size = max(1, math.ceil(len(mono) / max_waveform_points))
    starts = range(0, len(mono), frame_size)
    envelope = np.asarray([
        (float(np.min(mono[start:start + frame_size])),
         float(np.max(mono[start:start + frame_size])))
        for start in starts
    ])
    frame_times = (
        np.arange(len(envelope), dtype=float) * frame_size + frame_size / 2
    ) / sample_rate

    n_fft = 512
    hop_length = 256
    spectrum = np.abs(librosa.stft(
        mono,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    ))
    reference = float(np.max(spectrum))
    spectrum_db = librosa.amplitude_to_db(
        spectrum,
        ref=reference if reference > 0 else 1.0,
    )
    frame_indices = np.linspace(
        0,
        max(0, spectrum_db.shape[1] - 1),
        min(max_spectrogram_frames, spectrum_db.shape[1]),
        dtype=int,
    )
    return {
        "waveform_time_seconds": np.repeat(frame_times, 2),
        "waveform_amplitude": envelope.reshape(-1),
        "spectrogram_time_seconds": librosa.frames_to_time(
            frame_indices,
            sr=sample_rate,
            hop_length=hop_length,
        ),
        "spectrogram_frequency_hz": librosa.fft_frequencies(
            sr=sample_rate,
            n_fft=n_fft,
        ),
        "spectrogram_db": spectrum_db[:, frame_indices],
        "duration_seconds": duration,
        "sample_rate": int(sample_rate),
    }
