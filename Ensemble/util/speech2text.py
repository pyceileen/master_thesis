# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:00:27 2020

@author: Pei-yuChen
"""
import deepspeech
import numpy as np
import wave


model_file_path = r'C:\\Users\\Pei-yuChen\\Desktop\\Programming\\Model\\VocalSound\\deepspeech_model\\deepspeech-0.7.4-models.pbmm'
beam_width = 500
model = deepspeech.Model(model_file_path)
model.setBeamWidth(700)
model.enableExternalScorer(r'C:\\Users\\Pei-yuChen\\Desktop\\Programming\\Model\\VocalSound\\deepspeech_model\\deepspeech-0.7.4-models.scorer')

def speech2text(audio):
    w = wave.open(audio, 'r')
    frames = w.getnframes()
    buffer = w.readframes(frames)
    data16 = np.frombuffer(buffer, dtype=np.int16)    
    txt = model.stt(data16)
    return(txt)
