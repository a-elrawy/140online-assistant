# 140online Virtual Assistant 

This repository contains an Arabic Automatic Speech Recognition (ASR) model and a Faiss-indexing system using BERT text embeddings, designed as a virtual assistant for the 140online website, which contains data for over 200,000 companies. We utilize the NeMo framework for training the ASR model.

## Table of Contents
- [Arabic ASR Model](#arabic-asr-model)
- [Faiss-Indexing with BERT Text Embedding](#faiss-indexing-with-bert-text-embedding)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training the ASR Model](#training-the-asr-model)
- [Pretrained Weights](#pretrained-weights)
- [Language Modeling](#language-modeling)
- [Running Inference](#running-inference)
- [Running the Server](#running-the-server)


## Arabic ASR Model
Our ASR model is based on the Conformer architecture with a CTC (Connectionist Temporal Classification) loss. The model is trained using the NeMo framework, leveraging GPU acceleration for efficient training.

## Faiss-Indexing with BERT Text Embedding
We use BERT text embeddings to index and retrieve company information from the 140online website. The Faiss library is utilized to perform efficient similarity searches over the large dataset. For more detailed info check [Retrieval](/index/README.md)

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/a-elrawy/140online-assistant.git
    cd 140online-assistant
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure you have access to a GPU with CUDA installed.

## Dataset
To train the ASR model, you need to download the dataset and prepare the manifest files.

1. Download the dataset from [MGB-2 Arabic Speech Corpus](https://arabicspeech.org/resources/mgb2).

2. Crop the data to sentence length using Kaldi. Follow the instructions in the Kaldi recipe:
    ```sh
    git clone https://github.com/kaldi-asr/kaldi.git
    cd kaldi/egs/mgb2_arabic/s5
    ./local/mgb_data_prep.sh /path/to/mgb2/data
    ```

3. Create manifest files using the provided script in `data_utils`:
    ```sh
    python data_utils/prepare_manifest.py --mode 'train' --path /kaldi/mgb2arabic/
    ```
   This will generate `train_manifest.json`, change mode to generate `dev_manifest.json`, and `test_manifest.json` files.

## Training the ASR Model
To train the ASR model, run the following command:
```sh
python training/asr/asr_ctc/speech_to_text_ctc.py \
  --config-path="../conf/conformer/" \
  --config-name="conformer_ctc_char" \
  model.train_ds.manifest_filepath="manifest/train_manifest.json" \
  model.validation_ds.manifest_filepath="manifest/dev_manifest.json" \
  trainer.accelerator="gpu" \
  trainer.devices=2 \
  trainer.max_epochs=60
```
This command uses the Conformer CTC configuration and specifies the training and validation manifest files, as well as the GPU settings.

## Pretrained Weights
To run the ASR and retrieval system effectively, you need to download pre-trained model weights. We provide a script to simplify this process.



### Running the Script

1. Ensure you have the required permissions to execute the script:
    ```sh
    chmod +x scripts/download_weights.sh
    ```

2. Run the script to download the weights:
    ```sh
    scripts/download_weights.sh
    ```

This script will download the pre-trained weights and place them in the appropriate directories required by the system.



## Language Modeling
This section describes the process of building and utilizing language models (LM) for the 140online and MGB2 datasets using KenLM and SRILM. The language models are crucial for improving the accuracy of the ASR system. For more details check [language-modeling](lm/README.md)

## Running Inference
To run inference and test the ASR model, edit `test.py` as needed and run:
```sh
python test.py \
  --config-path="training/asr/conf/conformer/" \
  --config-name="conformer_ctc_char" \
  model.test_ds.manifest_filepath="manifest/test_manifest.json" \
  trainer.accelerator="gpu" \
  trainer.devices=1
```
This command specifies the test manifest file and GPU settings for inference.


## Running the Server
To start the server with the retrieval method, run:
```sh
python server.py
```
This command will launch the server, allowing you to use the virtual assistant to query the 140online company data.

### Endpoints

- **`/companies`**: Get a random sample of 100 companies.
- **`/transcribe`**: Transcribe audio from a given URL.
- **`/company_text`**: Retrieve company data based on text input.
- **`/company_audio`**: Retrieve company data based on audio input.

### Detaled API Usage
- **`/companies`**: 
  - **Method**: GET
  - **Description**: Retrieves a random sample of 100 companies from the database.
  - **Response**: A list of company records.

- **`/transcribe`**: 
  - **Method**: POST
  - **Description**: Transcribes audio from a given URL.
  - **Request Body**: JSON with `audio_url` field.
  - **Response**: A JSON object containing the transcription.

- **`/company_text`**: 
  - **Method**: POST
  - **Description**: Retrieves company data based on text input.
  - **Request Body**: Raw text of the company name.
  - **Response**: A JSON object containing the search results.

- **`/company_audio`**: 
  - **Method**: POST
  - **Description**: Retrieves company data based on audio input.
  - **Request Body**: JSON with `audio_url` field.
  - **Response**: A JSON object containing the transcription and search results.

This setup will enable you to interact with the virtual assistant and retrieve company information from the 140online database using both text and audio inputs.