from time import gmtime, strftime

import sounddevice as sd
import soundfile as sf
from keras.models import load_model

import CTC

model = load_model('model_20.keras', custom_objects={'CTCLoss': CTC.CTCLoss})

sample_rate = 16000
duration = 5
print("Speak now:")

# myrecording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
# audio_path = f'audio_{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}.wav'
# sd.wait()
#
# sf.write(audio_path, myrecording, samplerate=sample_rate)

frame_length = 256
frame_step = 160
fft_length = 384

encode = CTC.encode_single_sample("LJ001-0001.wav", frame_length, frame_step, fft_length)
encoded_predicted_text = model.predict(encode)
predicted_text = CTC.decode_batch_predictions(encoded_predicted_text)
print(predicted_text)
