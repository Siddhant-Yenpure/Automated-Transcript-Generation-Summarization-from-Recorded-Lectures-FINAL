from flask import Flask, render_template, request, redirect, url_for
import os
import ffmpeg
import whisper
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to extract audio from video
def extract_audio(video_path):
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_audio.wav")
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
    return audio_path

# Function to transcribe audio
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

# Function to generate summary
def generate_summary(text, num_sentences):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=num_sentences)
    return "\n".join(str(sentence) for sentence in summary)

# Route for the home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Route for handling uploads and processing
@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files or "lines" not in request.form:
        return redirect(url_for("index"))

    # Get the uploaded video and summary line count
    video = request.files["video"]
    num_lines = int(request.form["lines"])

    if video.filename == "":
        return redirect(url_for("index"))

    # Save the uploaded video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    # Process video
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)
    summary = generate_summary(transcript, num_lines)

    # Render the results page
    return render_template("result.html", transcript=transcript, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
