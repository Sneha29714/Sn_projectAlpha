import cv2
import pytesseract
from rapidfuzz import fuzz

# =========================================================
# OCR
# =========================================================


def extract_text_from_image(filepath):

    image = cv2.imread(filepath)

    if image is None:
        return ""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    _, thresh = cv2.threshold(resized, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh)

    return text


# =========================================================
# COMMON TEXT FUNCTIONS
# =========================================================


def normalize_text(text):

    if not text:
        return ""

    text = text.lower()

    text = " ".join(text.split())

    return text


def get_name_similarity(user_name, extracted_text):

    user_name = normalize_text(user_name)
    extracted_text = normalize_text(extracted_text)

    if not user_name or not extracted_text:
        return 0

    similarity = fuzz.partial_ratio(user_name, extracted_text)

    return similarity


# =========================================================
# RISK SCORE
# =========================================================


def calculate_risk_score(document_status, name_similarity,
                         document_format_valid):

    score = 0

    # Document number mismatch
    if "mismatch" in document_status.lower():

        score += 40

    # Document number not detected
    elif "not detected" in document_status.lower():

        score += 40

    # Name similarity
    if name_similarity < 85:

        score += 20

    # Invalid document format
    if not document_format_valid:

        score += 20

    return score


# =========================================================
# RISK LEVEL
# =========================================================


def get_risk_level(score):

    if score >= 50:

        return "High Risk"

    elif score >= 20:

        return "Medium Risk"

    return "Low Risk"


# =========================================================
# CENTRAL VERIFICATION ENGINE
# =========================================================


def verify_document(document_type, name, document_number, extracted_text):

    # =========================================
    # Name similarity
    # =========================================

    name_similarity = get_name_similarity(name, extracted_text)

    # =========================================
    # Initialize
    # =========================================

    extracted_document = None

    document_status = "Unknown"

    document_format_valid = False

    # =========================================
    # PAN
    # =========================================

    if document_type == "pan":

        from validators.pan import (extract_pan, validate_pan,
                                    validate_pan_format)

        extracted_document = extract_pan(extracted_text)

        document_status = validate_pan(document_number, extracted_document)

        document_format_valid = validate_pan_format(extracted_document)

    # =========================================
    # Aadhaar
    # =========================================

    elif document_type == "aadhaar":

        from validators.aadhaar import (extract_aadhaar, validate_aadhaar,
                                        validate_aadhaar_format)

        extracted_document = extract_aadhaar(extracted_text)

        document_status = validate_aadhaar(document_number, extracted_document)

        document_format_valid = validate_aadhaar_format(extracted_document)

    # =========================================
    # VISA
    # =========================================

    elif document_type == "visa":

        from validators.visa import (extract_visa_number, validate_visa,
                                     validate_visa_format)

        extracted_document = extract_visa_number(extracted_text)

        document_status = validate_visa(document_number, extracted_document)

        document_format_valid = validate_visa_format(extracted_document)

    # =========================================
    # Invalid document type
    # =========================================

    else:

        return {"success": False, "error": "Invalid document type"}

    # =========================================
    # Determine whether number matches
    # =========================================

    number_match = ("match" in document_status.lower()
                    and "mismatch" not in document_status.lower())

    # =========================================
    # Risk calculation
    # =========================================

    risk_score = calculate_risk_score(document_status, name_similarity,
                                      document_format_valid)

    risk_level = get_risk_level(risk_score)

    # =========================================
    # Structured result
    # =========================================

    return {
        "success": True,
        "document": {
            "type": document_type,
            "entered_number": document_number,
            "detected_number": extracted_document,
            "number_match": number_match,
            "format_valid": document_format_valid
        },
        "identity": {
            "name_similarity": name_similarity
        },
        "risk": {
            "score": risk_score,
            "level": risk_level
        }
    }


# =========================================================
# COMPLETE DOCUMENT PROCESSING
# =========================================================


def process_document(document_type, name, document_number, filepath):

    # -----------------------------------------
    # OCR
    # -----------------------------------------

    extracted_text = extract_text_from_image(filepath)

    if not extracted_text or not extracted_text.strip():

        return {
            "success": False,
            "error": "Could not extract text from document"
        }

    # -----------------------------------------
    # Verification
    # -----------------------------------------

    result = verify_document(document_type, name, document_number,
                             extracted_text)

    return result


def calculate_combined_risk(rule_risk, ai_risk):
    """
    Combine rule-based verification risk and AI visual risk.

    Rule-based checks are given more weight because they are
    deterministic validation checks.
    """

    combined_score = (rule_risk * 0.6) + (ai_risk * 0.4)

    return round(combined_score)


def get_combined_risk_level(score):
    if score >= 50:
        return "High Risk"
    elif score >= 20:
        return "Medium Risk"

    return "Low Risk"