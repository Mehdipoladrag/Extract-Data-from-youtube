import os
import json

audio_folder = "C:/Users/YOO/Desktop/O/audio"
output_file = "C:/Users/YOO/Desktop/O/audio_fine_tune.jsonl"

class FineTune:
    def __init__(self, audio_folder, output_file):
        self.audio_folder = audio_folder
        self.output_file = output_file

    def create_fine_tune_file(self):
        if not os.path.exists(self.audio_folder):
            os.makedirs(self.audio_folder)
  
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def prepare_samples(self):
        for file in os.listdir(self.audio_folder):
            if file.endswith((".mp3", ".wav", ".m4a")):
                audio_path = os.path.join(self.audio_folder, file)
                transcription_placeholder = ""  
                sample = {"audio_url": audio_path, "transcription": transcription_placeholder}
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

fine_tune = FineTune(audio_folder, output_file)
fine_tune.create_fine_tune_file()
fine_tune.prepare_samples()

print(f"Fine tune ===> {output_file}")
