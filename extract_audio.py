import os
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
import pandas as pd

def extract_audio(in_dir, out_dir, filename, model, processor):
    speech_array, _ = librosa.load(os.path.join(in_dir, filename), sr=16_000)

    features = processor(speech_array, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values
    attention_mask = features.attention_mask

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask)
    
    feat = logits["extract_features"]

    df_aux = pd.DataFrame(feat.numpy().reshape(-1, 512))

    df_aux.to_csv(os.path.join(out_dir, filename.rsplit(".")[0]+"."+filename.rsplit(".")[1]+".csv"), sep=";", index=False, header=True)


if __name__ == '__main__':
    audios_dir = "data/train/audios_16kHz/"
    audios_features_dir = "data/train/audios_features/"

    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)

    for audio in os.listdir(audios_dir):
        print(f'Processing {audio}')
        try:
            extract_audio(audios_dir, audios_features_dir, audio, model, processor)
        except:
            print(f"Error on file {audio}")
        