from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv

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

if __name__ == '__main__':
    app.run(debug=True)
