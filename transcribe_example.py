import whisper
model = whisper.load_model("base.en")
result1 = model.transcribe("LJ001-0075.wav", fp16=False, language='English')
result2 = model.transcribe("LJ001-0077.wav", fp16=False, language='English')