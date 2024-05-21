from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

audio_file_path = "eng_2min.wav"

waveform, sample_rate = torchaudio.load(audio_file_path)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

model_id = "facebook/mms-1b-fl102"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

inputs = processor(waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt")

with torch.no_grad():
    outputs = model(input_values=inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(outputs, dim=-1)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)

