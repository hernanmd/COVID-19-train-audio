# librosa feature extraction
import librosa
import numpy as np

# Run all the sound files get all the feature data and write it to
# a csv file

audio = "audio.wav"
y, sr = librosa.load(audio, mono=True, duration=1)
# features
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)

# printing the data
for feature in (chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc):
    print(np.mean(feature))

