import whisper

model = whisper.load_model("small")
result = model.transcribe('C:/Users/taitym/Downloads/jko.mp3', fp16=False)
print(result)