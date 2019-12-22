import io

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.oauth2 import service_account

creds = service_account.Credentials.from_service_account_file("/home/frank/Downloads/auth.json")
client = speech.SpeechClient(credentials=creds)

# Loads the audio into memory
with io.open("audio/Tongue twisters/canner_vincent.wav", 'rb') as audio_file:
 content = audio_file.read()
 audio = types.RecognitionAudio(content=content)

config = types.RecognitionConfig(
 encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
 sample_rate_hertz=44100,
 language_code='en-US')

# Detects speech in the audio file
response = client.recognize(config, audio)

for result in response.results:
 print('Transcript: {}'.format(result.alternatives[0].transcript))