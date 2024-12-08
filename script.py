#!/usr/bin/env python3

import sys
import json
from pydub import AudioSegment

from vosk import Model, KaldiRecognizer

CHANNELS=1
FRAME_RATE = 16000

model = Model(lang="ru")

# Large vocabulary free form recognition
rec = KaldiRecognizer(model, 16000)
rec.SetWords(True)

# You can also specify the possible word list
#rec = KaldiRecognizer(model, 16000, "zero oh one two three four five six seven eight nine")

mp3 = AudioSegment.from_mp3('C:/Users/taitym/Downloads/result_audio.wav')
mp3 = mp3.set_channels(CHANNELS)
mp3 = mp3.set_frame_rate(FRAME_RATE)

rec.AcceptWaveform(mp3.raw_data)
result = rec.Result()
text = json.loads(result)["text"]
print(result)

def get_text(path_to_audio_file: str, model: str = 'ru'):
    CHANNELS=1
    FRAME_RATE = 16000

    model = Model(lang=model)

    # Large vocabulary free form recognition
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    mp3 = AudioSegment.from_mp3(path_to_audio_file)
    mp3 = mp3.set_channels(CHANNELS)
    mp3 = mp3.set_frame_rate(FRAME_RATE)

    rec.AcceptWaveform(mp3.raw_data)
    result = rec.Result()
    result = json.loads(result)["text"]

    return result



with open('C:/Users/taitym/Downloads/text_example_en.mp3', "rb") as wf:
    wf.read(44) # skip header

    data = wf.read()
    #print(data)
    
    if rec.AcceptWaveform(data):
        res = json.loads(rec.Result())
    #print(res["text"])

    res = json.loads(rec.FinalResult())
    #print('привет')
    #print(res)
    #print(res["text"])