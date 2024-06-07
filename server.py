import os
import time
import shutil

from index import SimilarityFinder
from buckwalter import fromBuckWalter

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import aiohttp

from pyctcdecode import build_ctcdecoder
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from omegaconf import DictConfig, OmegaConf, open_dict
import nemo.collections.asr as nemo_asr



# Load the model and configure decoding
def load_model():
    # ctc_decoding = CTCDecodingConfig()
    # ctc_decoding.strategy = 'pyctcdecode'
    # ctc_decoding.beam.kenlm_path = 'lm_filtered.arpa'
    # ctc_decoding.beam.beam_size = 100
    # ctc_decoding.beam.beam_alpha = 0.5
    # ctc_decoding.beam.beam_beta = 1.5

    # decoding_cls = OmegaConf.structured(CTCDecodingConfig)
    # decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
    # decoding_cfg = OmegaConf.merge(decoding_cls, ctc_decoding)

    path = "nemo_experiments/Conformer-CTC-Char/2024-03-23_16-27-39/checkpoints/Conformer-CTC-Char--val_wer=0.2807-epoch=55-last.ckpt"
    asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(path)

    # with open_dict(asr_model.cfg):
    #     asr_model.cfg.decoding = decoding_cfg

    # asr_model.decoding = CTCDecoding(
    #     decoding_cfg=asr_model.cfg.decoding, vocabulary=asr_model.decoding.vocabulary)

    return asr_model


def download_file_from_firebase(storage_url: str, filename: str) -> None:
    response = requests.get(storage_url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Failed to download file from {storage_url}")


app = FastAPI()

model = load_model()

database_file = "140online/company_names.txt"  # Change this to your file path
similarity_finder = SimilarityFinder(database_file)


database = pd.read_csv('140online/compaines_data.csv')
database.fillna('', inplace=True)

@app.get('/companies')
def get_companies():
    # Get a random sample of 100 companies from the DataFrame
    sample_size = min(100, len(database))
    random_companies = database.sample(n=sample_size).to_dict(orient='records')
    return random_companies

@app.post('/transcribe')
async def transcribe_audio(audio_url: str):
    try:
        await download_file(audio_url, "uploaded_audio.wav")
        transcription = fromBuckWalter(model.transcribe(["uploaded_audio.wav"])[0])

        return {"transcription": transcription}
        
    except Exception as e:
        return {"error": str(e)}

@app.post('/company_text')
async def get_company_by_text(request: Request):
    try:
        text = await request.body()
        company_name = text.decode("utf-8")
        nearest_neighbors = similarity_finder.find_nearest_neighbors(company_name)
        search_result = []
        for nn in nearest_neighbors:
            search_result.append(database.query("Name == @nn").to_dict(orient='records')[0])
        return {
                "result": search_result}

    except Exception as e:
        return {"error": str(e)}


@app.post('/company_audio')
async def get_company_by_audio(audio_url: str):
    try:
        await download_file(audio_url, "uploaded_audio.wav")
        transcription = fromBuckWalter(model.transcribe(["uploaded_audio.wav"])[0])
        
        most_similar = similarity_finder.find_most_similar(transcription)

        search_result = database.query("Name == @most_similar").to_dict(orient='records')

        return {"transcription": most_similar,
                "result": search_result}
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    from pyngrok import ngrok
    public_url = ngrok.connect(8000).public_url
    print(public_url)
    uvicorn.run(app, host="0.0.0.0", port=8000)
