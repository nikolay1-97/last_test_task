from pyannote.audio import Pipeline
from pydub import AudioSegment
from get_gender import get_gender
from script import get_text

rec = AudioSegment.from_wav("C:/Users/taitym/Downloads/result_audio2.wav")

pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='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')
DEMO_FILE = {'uri': 'blabal', 'audio': "C:/Users/taitym/Downloads/result_audio2.wav"}
dz = pipeline(DEMO_FILE)

data = {}


#print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")
#print(type(dz))
#print(str(dz))
with open("C:/Users/taitym/Desktop/Python/audio_in_text/diarization.txt", "w") as text_file:
    text_file.write(str(dz))

for turn, _, speaker in dz.itertracks(yield_label=True):
    if speaker not in data:
        data[speaker] = {'segments':[{
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker,
            'text': get_text(rec[turn.start * 1000: turn.end * 1000].export(format='mp3')),
        }]}
        #[{'start': turn.start, 'end': turn.end}]
    else:
        data[speaker]['segments'].append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker,
            'text': get_text(rec[turn.start * 1000: turn.end * 1000].export(format='mp3')),
        })
        
    #tm = str(turn[0]).strip().split('-->')
    #v = tm[0].strip()[1:]
    #print(tm)
    #print(v)

#print(data)

for speaker in data.keys():
    start = data[speaker]['segments'][0]['start'] * 1000
    end = data[speaker]['segments'][0]['end'] * 1000
    audio_of_speaker = rec[start:end]

    if len(data[speaker]['segments']) > 1:
        for number_segments in range(1, len(data[speaker]['segments'])):
            start = data[speaker]['segments'][number_segments]['start'] * 1000
            end = data[speaker]['segments'][number_segments]['end'] * 1000
            audio_of_speaker += rec[start:end]
    data[speaker]['gender'] = get_gender(audio_of_speaker.export(format='wav'))

#r = rec[data['SPEAKER_01'][0]['start']*1000:data['SPEAKER_01'][0]['end']*1000]
#r.export("C:/Users/taitym/Downloads/first_seconds.wav", format='wav')
#print(get_gender("C:/Users/taitym/Downloads/first_seconds.wav"))
#data['SPEAKER_00']['audio_of_speaker'].export("C:/Users/taitym/Downloads/first_seconds.mp3", format='mp3')
#print(get_gender(data['SPEAKER_00']['audio_of_speaker'].export(format='wav')))
#r = get_text(data['SPEAKER_00']['audio_of_speaker'].export(format='mp3'))
#print(r)
res = []
for speaker in data.keys():
    for segment in data[speaker]['segments']:
        segment['gender'] = data[speaker]['gender']
        res.append(segment)

res.sort(key=lambda row: row['start'])
for i in res:
    print(i)

