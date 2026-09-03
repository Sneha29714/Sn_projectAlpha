import base64
import json

import ollama


# Local Ollama server
OLLAMA_HOST = "http://127.0.0.1:11434"

# Local Gemma 3 model
MODEL = "gemma3:4b"

client = ollama.Client(host=OLLAMA_HOST)


def encode_image(filepath):
    """
    Convert an image file into Base64.
    """

    with open(filepath, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def analyze_document_with_ai(filepath, document_type):
    """
    Analyze a document image using local Ollama + Gemma 3 4B.

    Returns:
        dict containing:
        - status
        - suspicious
        - confidence
        - risk_score
        - reasons
    """

    try:
        image_base64 = encode_image(filepath)

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

        response = client.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }
            ],
            format="json",
        )

        result_text = response["message"]["content"].strip()

        try:
            result = json.loads(result_text)

        except json.JSONDecodeError:
            return {
                "status": "available",
                "suspicious": None,
                "confidence": 0,
                "risk_score": 0,
                "reasons": [
                    "AI returned an invalid response format"
                ],
            }

        suspicious = result.get("suspicious", False)

        try:
            confidence = float(
                result.get("confidence", 0)
            )
        except (TypeError, ValueError):
            confidence = 0

        try:
            risk_score = int(
                result.get("risk_score", 0)
            )
        except (TypeError, ValueError):
            risk_score = 0

        confidence = max(
            0,
            min(1, confidence)
        )

        risk_score = max(
            0,
            min(100, risk_score)
        )

        reasons = result.get("reasons", [])

        if not isinstance(reasons, list):
            reasons = [str(reasons)]

        reasons = [
            str(reason)
            for reason in reasons
        ]

        return {
            "status": "available",
            "suspicious": bool(suspicious),
            "confidence": confidence,
            "risk_score": risk_score,
            "reasons": reasons,
        }

    except Exception as error:
        print(f"Local AI error: {error}")

        return {
            "status": "unavailable",
            "suspicious": None,
            "confidence": 0,
            "risk_score": 0,
            "reasons": [
                "Local AI analysis is unavailable"
            ],
        }