import argparse
import json

def parse_text_file(text_file, output_file, mode):
    entries = []
    with open(text_file, 'r', encoding='utf-8') as file:
      with open(output_file, 'w') as outfile:
        for line in file:
            parts = line.strip().split(' ')
            audio_filepath = f"{path}/segmented_wav/{mode}/" + parts[0] + ".wav"
            text = ' '.join(parts[1:])
            duration = (int(parts[0].split('-')[-1].split(':')[1]) - int(parts[0].split('-')[-1].split(':')[0])) / 100.0
            entry = {
                "audio_filepath": audio_filepath,
                "text": text,
                "duration": duration
            }
            json.dump(entry, outfile, ensure_ascii = False)
            outfile.write('\n')


def main(mode, path):
    dir = 'data'
    if mode == 'test':
       dir = 'DB'
    input_file = f"{path}/s5/{dir}/{mode}/text"  # Update this with the path to your input text file
    output_file = f"{mode}_manifest.json"  # Update this with the desired output JSON file path
    entries = parse_text_file(input_file, output_file, mode)
    print("JSON file has been created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Kaldi data.")
    parser.add_argument("mode", choices=["dev", "test", "train_mer80"], help="Specify the mode: 'dev', 'test', or 'train'")
    parser.add_argument("path",  help="kaldi-mgb2 path'")

    args = parser.parse_args()
    main(args.mode, args.path)

