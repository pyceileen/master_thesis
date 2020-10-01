# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:15:41 2020

@author: Pei-yuChen
"""

import librosa
import numpy as np
import pandas as pd

def extract_mfcc(audio_vector, sr):
    mfcc = librosa.feature.mfcc(y=audio_vector, sr=sr,
                                n_mels=40, n_fft=512,n_mfcc=40,
                                win_length=int(0.025*16000), 
                                hop_length=int(0.01*16000),
                                fmin=300, fmax=8000
                                )
    mfcc = pd.DataFrame(mfcc.T)
    mfcc = mfcc.rename(columns=lambda x:'mfcc_'+ str(x))
    return(mfcc)

# logged mel filter banks energies
def extract_logmel(audio_vector, sr):
    mel = librosa.feature.melspectrogram(y=audio_vector, sr=sr,
                                         n_mels=40, n_fft=512, 
                                         win_length=int(0.025*16000),
                                         hop_length=int(0.01*16000),
                                         fmin=300, fmax=8000)
    logmel= librosa.power_to_db(mel, ref=np.max)
    logmel = pd.DataFrame(logmel.T)
    logmel = logmel.rename(columns=lambda x:'logmel_'+ str(x))
    return(logmel)

def extract_chroma(audio_vector, sr):
    stft = np.abs(librosa.stft(audio_vector))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma = pd.DataFrame(chroma.T)
    chroma = chroma.rename(columns=lambda x:'chromac_'+ str(x))
    return(chroma)

def extract_contrast(audio_vector, sr):
    stft = np.abs(librosa.stft(audio_vector))
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast = pd.DataFrame(contrast.T)
    contrast = contrast.rename(columns=lambda x:'contrast_'+ str(x))
    return(contrast)

def extract_tonnetz(audio_vector, sr):
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_vector), 
                                      sr=sr)
    tonnetz = pd.DataFrame(tonnetz.T)
    tonnetz = tonnetz.rename(columns=lambda x:'tonnetz_'+ str(x))
    return(tonnetz)

def extract_pitch_mag(audio_vector, sr):
    pitches, magnitudes = librosa.piptrack(y=audio_vector, sr=sr)
    pitches = pd.DataFrame(pitches.T)
    pitches = pitches.rename(columns=lambda x:'pitches_'+ str(x))
    magnitudes = pd.DataFrame(magnitudes.T)
    magnitudes = magnitudes.rename(columns=lambda x:'magnitudes_'+ str(x))
    return(pitches, magnitudes)

def extract_zc_rate(audio_vector):
    zc_rate = librosa.feature.zero_crossing_rate(y=audio_vector)
    zc_rate = pd.DataFrame(zc_rate.T)
    zc_rate = zc_rate.rename(columns=lambda x:'zc_rate_'+ str(x))
    return(zc_rate)

def extract_rolloff(audio_vector, sr):
    rolloff = librosa.feature.spectral_rolloff(y=audio_vector, sr=sr)
    rolloff = pd.DataFrame(rolloff.T)
    rolloff = rolloff.rename(columns=lambda x:'rolloff_'+ str(x))
    return(rolloff)

def extract_flux(audio_vector, sr):
    flux = librosa.onset.onset_strength(y=audio_vector, sr=sr)
    flux = pd.DataFrame(flux.T)
    flux = flux.rename(columns=lambda x:'flux_'+ str(x))
    return(flux)

def extract_rms(audio_vector):
    rms = librosa.feature.rms(y=audio_vector)[0]
    rms = pd.DataFrame(rms.T)
    rms = rms.rename(columns=lambda x:'rms_'+ str(x))
    return(rms)

# below don't need stats
def extract_signal(audio_vector):
    signal_mean = np.mean(abs(audio_vector))
    signal_std = np.std(abs(audio_vector))
    signal_max = np.max(abs(audio_vector))
    signal_min = np.min(abs(audio_vector))
    d = {'signal_mean': [signal_mean], 'signal_std': [signal_std],
         'signal_max': [signal_max], 'signal_min': [signal_min]}
    signal = pd.DataFrame(d)
    return(signal)

def extract_silence(audio_vector, sr):
    rms = librosa.feature.rms(y=audio_vector)[0]
    silence = 0
    for e in rms:
        if e <= 0.4 * np.mean(rms):
            silence += 1
    silence = silence/float(len(rms))
    silence = pd.DataFrame({'silence':[silence]})
    return(silence)

def extract_harmony(audio_vector):
    y_harmonic, y_percussive = librosa.effects.hpss(y=audio_vector)
    harmony = np.mean(y_harmonic*1000)# harmonic (scaled by 1000)
    harmony = pd.DataFrame({'harmony':[harmony]})
    return(harmony)

def extract_acf(audio_vector):
    cl = 0.45 * np.mean(abs(audio_vector))
    center_clipped = []
    for s in audio_vector:
        if s >= cl:
            center_clipped.append(s - cl)
        elif s <= -cl:
            center_clipped.append(s + cl)
        elif np.abs(s) < cl:
            center_clipped.append(0)
    acfs  = librosa.core.autocorrelate(np.array(center_clipped))
    acf_max = 1000*np.max(acfs)/len(acfs)  # auto_corr_max (scaled by 1000)
    acf_std = np.std(acfs)  # auto_corr_std
    d = {'acf_max': [acf_max], 'acf_std': [acf_std]}
    acf = pd.DataFrame(d)
    return(acf)

def get_stats(df): # get mean, std, min, max
    feature_mean = df.mean(axis = 0).to_frame().T
    feature_mean.columns = [str(col) + '_mean' for col in feature_mean.columns]
    
    feature_std = df.std(axis = 0).to_frame().T
    feature_std.columns = [str(col) + '_std' for col in feature_std.columns]

    feature_min = df.min(axis = 0).to_frame().T
    feature_min.columns = [str(col) + '_min' for col in feature_min.columns]
    
    feature_max = df.max(axis = 0).to_frame().T
    feature_max.columns = [str(col) + '_max' for col in feature_max.columns]
    return(feature_mean,feature_std,feature_min,feature_max)

def feature_extraction_all(audio_vector, sr = 44100):
    mfcc = extract_mfcc(audio_vector, sr) 
    logmel = extract_logmel(audio_vector, sr)
    chroma= extract_chroma(audio_vector, sr)
    contrast= extract_contrast(audio_vector, sr)
    tonnetz= extract_tonnetz(audio_vector, sr)
    pitches, magnitudes = extract_pitch_mag(audio_vector, sr)
    zc_rate= extract_zc_rate(audio_vector)
    rolloff= extract_rolloff(audio_vector, sr)
    flux= extract_flux(audio_vector, sr)
    rms= extract_rms(audio_vector)
    
    ext_features = pd.concat([mfcc,logmel,
                              chroma,contrast,tonnetz,
                              pitches,magnitudes,zc_rate,rolloff,
                              flux,rms], axis=1)
    
    feature_mean,feature_std,feature_min,feature_max = get_stats(ext_features)
    
    # these don't need the four stats
    signal = extract_signal(audio_vector)
    silence= extract_silence(audio_vector, sr)
    harmony= extract_harmony(audio_vector)
    acf = extract_acf(audio_vector)
    
    features_all = pd.concat([feature_mean,feature_std,feature_min,feature_max,
                              signal,silence,harmony,acf], axis=1)    
    
    return(features_all)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
