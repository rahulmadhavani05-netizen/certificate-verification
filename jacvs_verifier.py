import streamlit as st
import json
from PIL import Image
import io
import pytesseract
import hashlib
from pdf2image import convert_from_bytes
import pandas as pd
import cv2
import numpy as np
import re
from rapidfuzz import fuzz, process  # For fuzzy matching

# ---------------- TESSERACT PATH ----------------
import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------------- ENHANCED OCR ----------------
def preprocess_image(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.medianBlur(enhanced, 3)
    height, width = denoised.shape
    if height < 1000 or width < 1000:
        scale = 2.0
        denoised = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(thresh)

def process_certificate_ocr(image):
    try:
        processed = preprocess_image(image)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, config=custom_config, lang='eng')
        data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        ocr_confidence = np.mean(confidences) if confidences else 0

        # Regex extraction
        extracted_data = {"name": "", "roll_no": "", "cert_id": ""}
        name_match = re.search(r'(?:Name|नाम)[:\s]*([A-Za-z\s]+?)(?=\n|$|Roll|Certificate)', text, re.IGNORECASE | re.DOTALL)
        extracted_data["name"] = name_match.group(1).strip() if name_match else ""
        roll_match = re.search(r'(?:Roll\s*(?:Number|No)[:\s-]*)([A-Z0-9/]+)', text, re.IGNORECASE)
        extracted_data["roll_no"] = roll_match.group(1).strip() if roll_match else ""
        cert_match = re.search(r'(?:Certificate\s*(?:ID|No)|Cert\s*ID|ID[:\s]*)([A-Z0-9/]+)', text, re.IGNORECASE)
        extracted_data["cert_id"] = cert_match.group(1).strip() if cert_match else ""

        result = {"extracted_data": extracted_data, "ocr_confidence": ocr_confidence, "full_text": text}
        if st.session_state.get('debug', False):
            print(f"=== OCR DEBUG ===\nFull Text: {text[:200]}...\nExtracted: {extracted_data}\nConfidence: {ocr_confidence:.1f}%\n================")
        return result
    except Exception as e:
        return {"extracted_data": {}, "ocr_confidence": 0, "full_text": "", "error": str(e)}

# ---------------- TEXT NORMALIZATION ----------------
def normalize_text(text):
    return re.sub(r'[^\w]', '', text.lower().strip())

# ---------------- FUZZY MATCH ----------------
def fuzzy_lookup(name_norm, records_dict, threshold=85):
    best_match = process.extractOne(name_norm, records_dict.keys(), scorer=fuzz.ratio)
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    return None

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="JACVS Verifier", layout="wide")
st.title("🛡️ JACVS - Jharkhand Academic Credential Verification System")
st.markdown("Upload a certificate (PDF/JPG/PNG) and a CSV file for verification.")

with st.sidebar:
    st.header("How to Use")
    st.write("- Clear scans only")
    st.write("- Supported: PDF, JPG, JPEG, PNG")
    st.write("- CSV columns: name, roll_no, cert_id")
    st.write("- Institutions: contact admin for bulk tools")
    debug = st.checkbox("Enable Debug Mode")
    st.session_state.debug = debug

# ---------------- CSV UPLOAD ----------------
csv_file = st.file_uploader("Upload CSV file", type=['csv'])
records_dict = {}
if csv_file:
    try:
        records_df = pd.read_csv(csv_file)
        if all(col in records_df.columns for col in ['name', 'roll_no', 'cert_id']):
            records_dict = {normalize_text(row['name']): 
                            {"roll_no": normalize_text(row['roll_no']),
                             "cert_id": normalize_text(row['cert_id'])}
                            for _, row in records_df.iterrows()}
            st.success(f"Loaded {len(records_dict)} records")
            if debug:
                st.write("Sample Records:", list(records_dict.items())[:3])
        else:
            st.error("CSV must have columns: name, roll_no, cert_id")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# ---------------- CERTIFICATE UPLOAD ----------------
uploaded_file = st.file_uploader("Choose a certificate", type=['pdf', 'jpg', 'jpeg', 'png'])
if uploaded_file:
    file_content = uploaded_file.read()
    uploaded_file.seek(0)

    if uploaded_file.type == "application/pdf":
        try:
            images = convert_from_bytes(file_content)
            if len(images) == 0:
                st.error("No pages in PDF")
                st.stop()
        except Exception as e:
            st.error(f"PDF error: {e}")
            st.stop()
    else:
        images = [Image.open(io.BytesIO(file_content))]

    st.subheader("Uploaded Certificate Pages")
    for i, image in enumerate(images):
        st.image(image, caption=f"Page {i+1}", use_column_width=True)

    # OCR
    full_extracted_data = []
    for i, image in enumerate(images):
        with st.spinner(f"Processing page {i+1}..."):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            full_extracted_data.append(process_certificate_ocr(image))

    # Combine pages
    combined_data = {"name": "", "roll_no": "", "cert_id": "", "full_text": ""}
    for page in full_extracted_data:
        extracted = page.get("extracted_data", {})
        for key in ["name", "roll_no", "cert_id"]:
            if extracted.get(key) and not combined_data[key]:
                combined_data[key] = extracted[key]
        combined_data["full_text"] += page.get("full_text", "") + "\n"

    # Normalize extracted fields
    name_norm = normalize_text(combined_data.get("name", ""))
    roll_norm = normalize_text(combined_data.get("roll_no", ""))
    cert_norm = normalize_text(combined_data.get("cert_id", ""))

    if debug:
        st.write(f"Normalized Extracted: Name='{name_norm}', Roll='{roll_norm}', Cert='{cert_norm}'")

    # Document hash
    all_pages_bytes = io.BytesIO()
    for image in images:
        temp = io.BytesIO()
        image.save(temp, format='PNG')
        all_pages_bytes.write(temp.getvalue())
    document_hash = hashlib.sha256(all_pages_bytes.getvalue()).hexdigest()

    # ---------------- VERIFICATION ----------------
    anomalies = []
    avg_confidence = np.mean([p.get('ocr_confidence', 0) for p in full_extracted_data])
    confidence_score = int(avg_confidence)
    status = "Valid"
    recommendation = "Proceed with verification."

    if records_dict:
        matched_name = fuzzy_lookup(name_norm, records_dict)
        if matched_name:
            db_roll = records_dict[matched_name].get('roll_no', '')
            db_cert = records_dict[matched_name].get('cert_id', '')
            roll_score = fuzz.ratio(roll_norm, db_roll)
            cert_score = fuzz.ratio(cert_norm, db_cert)

            if roll_score < 85 or cert_score < 85:
                anomalies.append("Mismatch in Roll No or Certificate ID")
                status = "Caution"
                confidence_score = min(confidence_score, 60)
                recommendation = "Manual review recommended."
            else:
                status = "Valid"
                confidence_score = max(confidence_score, 90)
        else:
            anomalies.append("Name not found in records")
            status = "Forged"
            confidence_score = min(confidence_score, 30)
            recommendation = "Document appears invalid."
    else:
        st.warning("No CSV loaded. Using OCR only.")
        status = "OCR Processed"

    if avg_confidence < 70:
        anomalies.append("Low OCR confidence - improve scan quality")
        confidence_score = min(confidence_score, 30)
    if not anomalies and avg_confidence > 80:
        confidence_score = 95

    result = {
        "status": status,
        "confidence_score": confidence_score,
        "recommendation": recommendation,
        "anomalies": anomalies,
        "extracted_data": combined_data,
        "document_hash": document_hash,
        "full_text": combined_data["full_text"],
        "avg_ocr_confidence": avg_confidence
    }

    # ---------------- DISPLAY ----------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Verification Report")
        status_color = "🟢" if status=="Valid" else ("🟡" if status=="Caution" else "🔴")
        st.markdown(f"**Status:** {status_color} {status} ({confidence_score}% Confidence)")
        st.write("**Recommendation:**", recommendation)
        if anomalies:
            st.error("⚠️ Anomalies Detected:")
            for a in anomalies: st.write(f"- {a}")
        else:
            st.success("✅ No issues found.")

    with col2:
        st.subheader("📄 Extracted Data")
        for k,v in combined_data.items():
            if k != "full_text" and v:
                st.write(f"**{k.replace('_',' ').title()}:** {v}")
        st.write(f"**Document Hash:** {document_hash[:16]}...")
        if debug:
            st.text(result['full_text'][:200])

    report_json = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button("📥 Download Report (JSON)", report_json, file_name="jacvs_report.json", mime="application/json")

st.markdown("---")
st.markdown("Built for **Jharkhand Education** | **Privacy Notice:** No data stored without consent.")
