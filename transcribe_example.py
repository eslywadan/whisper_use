import whisper
import os
model = whisper.load_model("base")

def transcribe():
    result1 = model.transcribe("LJ001-0075.wav", fp16=False, language='English')
    result2 = model.transcribe("LJ001-0077.wav", fp16=False, language='English')
    result3 = model.transcribe("SuiSiann_0013.wav", fp16=False, language='zh')

def detect_lan_imtong():
    results = loop_audios('ImTong')
    return results

def loop_audios(filepath):
    filelist = os.listdir(filepath)
    detect_results = {}
    for file in filelist:
        if file.split(".")[1] == 'wav':
            audio = whisper.load_audio(f"./{filepath}/{file}")
            detect_result= detect_lan(audio)
            detect_results[file] = detect_result
    
    return detect_results

def single_audio(filepath):
    audio = whisper.load_audio(f"{filepath}")
    detect_result= detect_lan(audio)
    return detect_result

def detect_lan(audio, return_len=3):
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    sprobs = sorted(probs.items(), key=lambda x:x[1], reverse=True)
    return sprobs[0:return_len]

def decode_audio(mel):
    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    # print the recognized text
    print(result.text)
