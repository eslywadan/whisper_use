import whisper
model = whisper.load_model("base")
result1 = model.transcribe("LJ001-0075.wav", fp16=False, language='English')
result2 = model.transcribe("LJ001-0077.wav", fp16=False, language='English')

result3 = model.transcribe("SuiSiann_0013.wav", fp16=False, language='zh')







# Load the model
model = whisper.load_model("whisper/asr/wav2vec2-large-xlsr-53")
audio = whisper.load_audio("SuiSiann_0013.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
