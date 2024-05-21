import whisper

model = whisper.load_model("small")
result = model.transcribe('eng_2min.wav')
print(result["text"])
