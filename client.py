import requests

# Define the URL of the FastAPI server
SERVER_URL = "https://3867-196-204-183-233.ngrok-free.app/transcribe"

# Define the path to the audio file you want to transcribe
AUDIO_PATH = "test2r.wav"
files = {'audio_file': open(AUDIO_PATH, 'rb')}


# Send a POST request to the server with the audio path
import time
t = time.time()
response = requests.post(SERVER_URL, params={'audio_url': 'https://firebasestorage.googleapis.com/v0/b/graduationproject-5e803.appspot.com/o/audio%2Faudio_record_20240514_131415.wav?alt=media&token=480e3d0d-5d00-4b9e-8ad2-b2daf0b8c4a6'})

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract the transcription from the response
    result = response.json()
    transcription = result["transcription"]
    print("Transcription:", transcription)
    print("Time Taken: ", time.time()-t)
else:
    print("Error:", response.text)
