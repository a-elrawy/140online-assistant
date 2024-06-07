import os
import json
import wave
import contextlib

from buckwalter import toBuckWalter
# Define the paths
dataset_dir = ''
sentences_path = os.path.join(dataset_dir, 'sentences.txt')
audio_dir = os.path.join(dataset_dir, 'audio')
manifest_path = os.path.join(dataset_dir, '140_test_manifest.json')

# Read the sentences
with open(sentences_path, 'r', encoding='utf-8') as f:
    sentences = f.readlines()

# Create the manifest list
manifest = []

# Iterate over audio files
with open(manifest_path, 'w') as outfile:
    for index in range(0, 300):
        audio_filename = '/lfs01/workdirs/hlwn038u1/MGB2/' + audio_dir + f'/{index}.wav'
        text = sentences[index].replace('\n', '')
        text = toBuckWalter(text)

        with contextlib.closing(wave.open(audio_filename, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            
            # Define the record
        record = {
                "audio_filepath": audio_filename,
                "text": text,
                "duration": duration  # You can calculate or provide the duration if available
            }
        json.dump(record, outfile, ensure_ascii = False)
        outfile.write('\n')
            # Add the record to the manifest


print(f'Manifest file created at {manifest_path}')
