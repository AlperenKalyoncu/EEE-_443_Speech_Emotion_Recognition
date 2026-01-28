import torch
import torchaudio
import soundfile as sf

class MFCCExtractor:
    def __init__(self,
        n_mfcc: int = 13,
        target_sr: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 40,
        max_seconds: float = 3.0,
        normalize: bool = True
        ):

        self.n_mfcc = n_mfcc
        self.target_sr = target_sr
        self.max_samples = int(target_sr * max_seconds)
        self.normalize = normalize

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=target_sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
            },
        )

    def load_wav_mono_resample(self, path: str) -> torch.Tensor:
        data, sr = sf.read(path, dtype="float32")

        wav = torch.from_numpy(data)
        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        wav = wav.unsqueeze(0)

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        return wav

    def pad_or_truncate(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[1]
        if n > self.max_samples:
            return wav[:, :self.max_samples]
        if n < self.max_samples:
            return torch.nn.functional.pad(wav, (0, self.max_samples - n))
        return wav

    def __call__(self, path: str) -> torch.Tensor:
        wav = self.load_wav_mono_resample(path)
        wav = self.pad_or_truncate(wav)

        mfcc = self.mfcc_transform(wav)
        mfcc = mfcc.squeeze(0).transpose(0, 1)

        if self.normalize:
            mfcc = (mfcc - mfcc.mean(dim=0)) / (mfcc.std(dim=0) + 1e-6)

        return mfcc

def audio_duration(path: str) -> float:
    data, sr = sf.read(path, dtype="float32")
    return len(data) / sr