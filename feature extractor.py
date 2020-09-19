#librosa feature extraction
import librosa
import numpy as np

#Run all the sound files get all the feature data and write it to
#a csv file

audio = 'F://ml//projects//soundy numpy//ramana-test.wav' 
y , sr = librosa.load(audio, mono=True, duration=1)
#features
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)

#printing the data
print(np.mean(chroma_stft))
print(".................................")

print(np.mean(spec_cent))
print(".................................")

print(np.mean(spec_bw))
print(".................................")

print(np.mean(rolloff))
print(".................................")

print(np.mean(zcr))
print(".................................")

print(np.mean(mfcc))
print(".................................")