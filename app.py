import openai
import os
import requests
import whisper
import tempfile
from dotenv import load_dotenv
from flask import Flask, request, jsonify


app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

load_dotenv()  # loads variables from .env file


def analyze_text_possibility_and_reason(text):
    prompt = (
        "Based on the following conversation transcript, determine if there is a chance the customer might become a lead in the future. "
        "Respond ONLY with two lines:\n"
        "1) Possibility: True or Possibility: False\n"
        "2) Reason: A concise reason (1-2 sentences) explaining your decision.\n\n"
        "Transcript:\n"
        f"{text}\n"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise and concise assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def parse_possibility_and_reason(output_text):
    lines = output_text.splitlines()
    possibility = None
    reason = None
    for line in lines:
        if line.lower().startswith("possibility:"):
            val = line.split(":", 1)[1].strip().lower()
            if val == "true":
                possibility = True
            elif val == "false":
                possibility = False
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    return possibility, reason



###############################
# Part 1: Audio Transcription #
###############################

def download_audio_from_url(url):
    """Downloads an audio file from a URL and returns its temporary file path."""
    local_filename = url.split("/")[-1]
    file_ext = os.path.splitext(local_filename)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    print(f"Downloading audio from {url} to temporary file {temp_file.name}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        raise Exception(f"Failed to download file from {url}. Status code: {response.status_code}")
    return temp_file.name

def generate_transcription_with_timestamps(result):
    """Generates a single text that includes both transcription and timestamps."""
    transcription_with_timestamps = ""
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        transcription_with_timestamps += f"[{start_time:.2f} - {end_time:.2f}] {text}\n"
    return transcription_with_timestamps

def transcribe_audio(file_path):
    """Transcribes an audio file using Whisper and returns the transcription with timestamps in a single text."""
    print(f"Transcribing audio file: {file_path}")
    model = whisper.load_model("tiny")
    result = model.transcribe(file_path, task="translate", verbose=False)
    
    # Generate transcription with timestamps combined
    transcription_with_timestamps = generate_transcription_with_timestamps(result)
    
    return transcription_with_timestamps


##########################################
# Original Main Execution (Unchanged)    #
##########################################


@app.route('/analyze', methods=['POST'])
def analyze():
    # Read raw text from the body
    text = request.data.decode('utf-8').strip()
    if not text:
        return jsonify({"error": "Empty request body"}), 400

    try:
        output = analyze_text_possibility_and_reason(text)
        possibility, reason = parse_possibility_and_reason(output)
        if possibility is None or reason is None:
            return jsonify({"error": "Could not parse model output"}), 500
        return jsonify({
            "possibility": possibility,
            "reason": reason
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##################################
# API Endpoint for Transcription #
##################################

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    API endpoint that accepts a POST request with JSON body containing a "url" key.
    Returns a JSON object with 'transcription_with_timestamps' as a single text.
    """
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing URL parameter in JSON body'}), 400
    
    audio_url = data['url']
    
    try:
        # Step 1: Download the audio file
        audio_file_path = download_audio_from_url(audio_url)
        
        # Step 2: Transcribe the downloaded audio file
        transcription_with_timestamps = transcribe_audio(audio_file_path)
        
        # Return the combined transcription with timestamps as a JSON object
        return jsonify({'transcription_with_timestamps': transcription_with_timestamps})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return 'Flask app is running on azure...!'

if __name__ == '__main__':
    app.run(debug=True)
