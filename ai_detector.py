import base64
import json
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not configured in .env")


client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


MODEL = "google/gemma-3-4b-it"


def encode_image(filepath):
    """
    Convert an image file into Base64.
    """

    with open(filepath, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def get_mime_type(filepath):
    """
    Determine the MIME type of the uploaded image.
    """

    extension = os.path.splitext(filepath)[1].lower()

    if extension in [".jpg", ".jpeg"]:
        return "image/jpeg"

    if extension == ".png":
        return "image/png"

    raise ValueError("Only JPG, JPEG and PNG images are supported")


def analyze_document_with_ai(filepath, document_type):
    """
    Analyze a document image using Qwen3-VL.

    Returns:
        dict containing:
        - suspicious
        - confidence
        - risk_score
        - reasons
    """

    image_base64 = encode_image(filepath)
    mime_type = get_mime_type(filepath)

    prompt = f"""
You are an AI-assisted document screening system.

Analyze this {document_type} document image for visible signs
that could indicate manipulation, alteration, or suspicious editing.

Look carefully for:

- overwritten or altered text
- handwritten text over printed text
- inconsistent fonts
- inconsistent font sizes
- unusual alignment
- unusual spacing
- duplicated elements
- inconsistent colors
- suspicious borders or backgrounds
- image editing artifacts
- pasted or replaced areas
- unusual positioning of document elements
- distorted or suspicious QR/barcode areas
- inconsistent visual quality between different parts of the document

IMPORTANT:

You are performing visual anomaly analysis only.

Do NOT claim with certainty that the document is genuine or forged.
A suspicious result means that visible characteristics deserve further
verification.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "suspicious": true,
    "confidence": 0.85,
    "risk_score": 75,
    "reasons": [
        "Visible handwritten text appears over a printed field",
        "Possible inconsistent formatting around the altered area"
    ]
}}

Rules:

- suspicious must be true or false
- confidence must be a number between 0 and 1
- risk_score must be an integer from 0 to 100
- reasons must be a JSON array of short strings
- Do not include markdown
- Do not include explanations outside the JSON
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,"
                                f"{image_base64}"
                            )
                        }
                    }
                ]
            }
        ]
    )

    result_text = response.choices[0].message.content.strip()

    # Sometimes models wrap JSON in ```json ... ```
    if result_text.startswith("```"):
        result_text = result_text.replace("```json", "")
        result_text = result_text.replace("```", "")
        result_text = result_text.strip()

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        return {
            "suspicious": False,
            "confidence": 0,
            "risk_score": 0,
            "reasons": [
                "AI returned an invalid response format"
            ]
        }

    # Basic safety checks on AI output
    suspicious = bool(result.get("suspicious", False))

    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0

    try:
        risk_score = int(result.get("risk_score", 0))
    except (TypeError, ValueError):
        risk_score = 0

    confidence = max(0, min(1, confidence))
    risk_score = max(0, min(100, risk_score))

    reasons = result.get("reasons", [])

    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    reasons = [
        str(reason)
        for reason in reasons
    ]

    return {
        "suspicious": suspicious,
        "confidence": confidence,
        "risk_score": risk_score,
        "reasons": reasons
    }