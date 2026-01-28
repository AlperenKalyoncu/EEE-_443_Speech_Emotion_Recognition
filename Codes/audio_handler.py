import os
import glob
import mfcc_extractor as sound
import numpy as np
import soundfile as sf

DEFAULT_DIR = r"AudioWAV"
NOISE_LEVEL = 0.005
CROP_LENGTH = 3.0

N_SHIFTS = 1
N_SPEEDS = 1
ADD_NOISE_PROB = 0.5

MAX_SHIFT_MS = 1000 
SPEED_FACTORS = (0.9, 1.1)

def get_emotion_distribution(audioFiles):
    EMOTIONS = {
        "ANG": 0,  # anger
        "DIS": 1,  # disgust
        "FEA": 2,  # fear
        "HAP": 3,  # happy
        "NEU": 4,  # neutral
        "SAD": 5,  # sadness
    }

    print("number of audio files: ", len(audioFiles))

    labels = [EMOTIONS[audioFile.split("_")[2]] for audioFile in audioFiles]

    no_of_labels = [0,0,0,0,0,0]

    for label in labels:
        no_of_labels[label] += 1

    return labels, no_of_labels

def time_shift(x: np.ndarray, sr: int, max_shift_ms: int) -> np.ndarray:

    max_shift = int(sr * max_shift_ms / 1000.0)

    shift = np.random.randint(-max_shift, max_shift + 1)

    if shift < 0:
        return x[-shift:]
    
    return x

def speed_change(x: np.ndarray, factor: float) -> np.ndarray:
    n = x.shape[0]

    if n < 2:
        return x

    new_n = max(2, int(n / factor))
    old_idx = np.arange(n, dtype=np.float32)
    new_idx = np.linspace(0, n - 1, new_n, dtype=np.float32)
    y = np.interp(new_idx, old_idx, x).astype(np.float32)
    return y

def add_noise(x: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.normal(0.0, noise_level, size=x.shape).astype(np.float32)
    return np.clip(x + noise, -1.0, 1.0)

def create_modified_directory(audioDirectory):

    os.makedirs(audioDirectory)

    if not os.path.isdir(DEFAULT_DIR):
        print(f"Directory {DEFAULT_DIR} does not exist")
        return False
    
    audioFiles = sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.wav")))

    for wav_path in audioFiles:
        data, sr = sf.read(wav_path, dtype="float32")
        data = np.asarray(data, dtype=np.float32)

        if data.ndim == 2:
            data = data.mean(axis=1)

        base = os.path.splitext(os.path.basename(wav_path))[0]

        base_clip = np.clip(data, -1.0, 1.0)

        out_base = os.path.join(audioDirectory, f"{base}_base.wav")
        sf.write(out_base, base_clip, sr)

        for s in range(N_SHIFTS):
            x = time_shift(base_clip, sr, MAX_SHIFT_MS)
            x = add_noise(x, NOISE_LEVEL)

            out_path = os.path.join(audioDirectory, f"{base}_shift{s}.wav")
            sf.write(out_path, x, sr)

        for t in range(N_SPEEDS):
            factor = np.random.uniform(SPEED_FACTORS[0], SPEED_FACTORS[1])
            x = speed_change(base_clip, factor)

            x = add_noise(x, NOISE_LEVEL)   

            out_path = os.path.join(audioDirectory, f"{base}_speed{t}.wav")
            sf.write(out_path, x.astype(np.float32), sr)

def get_file_info(audioDirectory = r"AudioWAVModified", n_mfccs = 40):

    if not os.path.isdir(audioDirectory):
        create_modified_directory(audioDirectory)
       
    audioFiles = sorted(glob.glob(os.path.join(audioDirectory, "*.wav")))

    labels, no_of_labels = get_emotion_distribution(audioFiles)

    print("ANG:", no_of_labels[0])
    print("DIS:", no_of_labels[1])
    print("FEA:", no_of_labels[2])
    print("HAP:", no_of_labels[3])
    print("NEU:", no_of_labels[4])
    print("SAD:", no_of_labels[5])

    mfcc = sound.MFCCExtractor(n_mfcc = n_mfccs)
    mfccs = [mfcc(audioFile) for audioFile in audioFiles]

    return labels, mfccs
    