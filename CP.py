import ffmpeg
import whisper

# Step 1: Extract audio from video
def extract_audio(video_file):
    audio_file = "upload/extracted_audio.wav"
    ffmpeg.input(video_file).output(audio_file).run(overwrite_output=True)
    return audio_file

# Step 2: Transcribe audio
def transcribe_audio(audio_file):
    model = whisper.load_model("base")  # You can choose 'tiny', 'small', 'medium', 'large'
    result = model.transcribe(audio_file)
    return result['text']

# Step 3: Save transcript to file
def save_transcript(transcript, output_file):
    with open(output_file, 'w') as f:
        f.write(transcript)
