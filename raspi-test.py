import pyaudio
from tensorflow import keras
import pickle
import librosa
import numpy as np
import time
import websocket
import base64
import json
from threading import Thread
import wave
import os

""" sklearn version must match: pip install scikit-learn==1.0.2 """
""" Get the deivice index from mic-array-setup/get_device_index.py """

keras_model_from_json = keras.models.model_from_json
API = "TYPE YOUR Assembly API"  #Assembly real-time transcription API

# Pyaudio Constant
CHUNK = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 44100
FILENAME = "recorded_audio_0.wav"
p = pyaudio.PyAudio()
frames = []
record_length = int(SAMPLE_RATE / CHUNK * 6)  # 6 seconds
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

def delete_pre_files():
    if os.path.exists(FILENAME):
        os.remove(FILENAME)
    else:
        print("You are good to go!")

# Load all models and encoders
def load_all():
    # Model
    start_time = time.time()  # Record the start time

    json_file = open('model/CNN_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model/best_model1_weights.h5")
    # Scaler and Encoder
    with open('model/scaler2.pickle', 'rb') as f:
        scaler2 = pickle.load(f)
    with open('model/encoder2.pickle', 'rb') as f:
        encoder2 = pickle.load(f)

    end_time = time.time()  # Record the end time
    # Print the time taken
    print(f'Time taken to load model: {end_time - start_time:.2f} seconds')

    return loaded_model, scaler2, encoder2

# Feature extraction functions
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result

def get_predict_feat(path, scaler2):
    d, sr = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res).reshape(1, 2376)
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

def prediction(path, model, encoder, scaler):
    print("\n Start processing...")
    res = get_predict_feat(path, scaler)
    predictions = model.predict(res)
    y_predict = encoder.inverse_transform(predictions)
    print(y_predict[0][0])

def on_message(ws, message):
    transcript = json.loads(message)
    text = transcript['text']
    print(text)

def on_open(ws):

    def send_data():
        global frames
        # global model, scaler, encoder
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            data = base64.b64encode(data).decode("utf-8")
            json_data = json.dumps({"audio_data": str(data)})
            ws.send(json_data)

            if len(frames) >= record_length:
                path = save_files(FILENAME, frames)
                frames = frames[record_length:]
                # prediction(path, model, encoder, scaler)
                Thread(target=prediction, args=(path, model, encoder, scaler)).start()

    Thread(target=send_data).start()

def on_error(ws, error):
    print(error)

def on_close(ws, reason, code):
    stream.close()
    stream.stop_stream()
    p.terminate()
    print("WebSocket closed")

def save_files(file, recorded_frames):
    with wave.open(file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(recorded_frames))

    return file

def main():

    delete_pre_files()
    # Load all models and encoders at first
    global model, scaler, encoder  # make these global
    model, scaler, encoder = load_all()

    # Start the WebSocket connection
    websocket.enableTrace(False)
    auth_header = {"Authorization": API}

    ws = websocket.WebSocketApp(
        f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={SAMPLE_RATE}",
        header=auth_header,
        on_message=on_message,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close)

    ws.run_forever()

# Call main function
if __name__ == "__main__":
    main()

