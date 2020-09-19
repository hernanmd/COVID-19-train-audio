#-----------------------------------------------------------------------------------------------------------------------------#
#                                   Listening to audio files and working out on classifying 
#-----------------------------------------------------------------------------------------------------------------------------#

import librosa
import numpy as np

audio = 'F://ml//projects//soundy numpy//soundy numpy//roshan-test.wav' 
y , sr = librosa.load(audio, mono=True, duration=1)

#features
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)

#loading up variables
chroma_stft = np.mean(chroma_stft)
spec_cent = np.mean(spec_cent)
spec_bw = np.mean(spec_bw)
rolloff = np.mean(rolloff)
zcr = np.mean(zcr)
mfcc = np.mean(mfcc)

#-----------------------------------------------------------------------------------------------------------------------------#
#                                                Classification starts here
#-----------------------------------------------------------------------------------------------------------------------------#

#importing required modules for Classification
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

dt = pd.read_csv('feature.csv')#the dataset
#print(dt.head(0))
train, test = train_test_split(dt)

#print(train.shape)
#print(test.shape)

train_X = train[['chroma_stft','spec_cent','spec_bw','rolloff','zcr','mfcc']]
train_y = train.prognosis

test_X = test[['chroma_stft','spec_cent','spec_bw','rolloff','zcr','mfcc']]
test_y = test.prognosis

#knn
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_X, train_y)

val = np.array([chroma_stft,spec_cent,spec_bw,rolloff,zcr,mfcc])
val=val.reshape(1,-1)
prediction = model.predict(val)
print(prediction)
