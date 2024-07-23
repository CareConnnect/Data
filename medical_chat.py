import speech_recognition as sr
from pydub import AudioSegment
import os
from datasets import DataSet
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# 오디오 파일 경로
audio_folder_path = 'C:\Users\tjdwn\Downloads\download (1).tar\141.의료진_및_환자_음성\01.데이터\2.Validation\라벨링데이터\라벨링데이터.zip.part0'

def convert_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_folder_path)
    audio.export('temp.wav', format = 'wav')
    with sr.AudioFile('temp.wav') as source:
        audio_data = recognizer.record(source)
        text = recognizer.record(source)
    os.remove('temp.wav')
    return text

# 폴더 내 모든 오디오 파일 텍스트로 변환
audio_files = [f for f in os.listdir(audio_folder_path) if f.endswith('.wav')]
text = []
for audio_file in audio_files:
    audio_path = os.path.join(audio_folder_path)