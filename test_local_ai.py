from ai_detector import analyze_document_with_ai


image_path = r"C:\Users\Subangkar\Downloads\pan-card.webp"

result = analyze_document_with_ai(
    image_path,
    "PAN card"
)

print("\n===== LOCAL AI RESULT =====")
print(result)