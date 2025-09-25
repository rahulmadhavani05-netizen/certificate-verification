import streamlit as st
import pandas as pd
from PIL import Image
import io
import pytesseract
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import re
import hashlib
from rapidfuzz import fuzz, process
import json
import platform

# ---------------- Tesseract OCR path ----------------
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------------- Helper Functions ----------------
def preprocess_image(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.medianBlur(enhanced, 3)
    height, width = denoised.shape
    if height < 1000 or width < 1000:
        denoised = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(thresh)

def ocr_extract(image):
    processed = preprocess_image(image)
    text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6')
    data = pytesseract.image_to_data(processed, config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
    ocr_conf = np.mean(confidences) if confidences else 0

    extracted = {"name": "", "roll_no": "", "cert_id": ""}
    name_match = re.search(r'(?:Name|नाम)[:\s]*([A-Za-z\s]+?)(?=\n|$|Roll|Certificate)', text, re.IGNORECASE | re.DOTALL)
    extracted["name"] = name_match.group(1).strip() if name_match else ""
    roll_match = re.search(r'(?:Roll\s*(?:Number|No)[:\s-]*)([A-Z0-9/]+)', text, re.IGNORECASE)
    extracted["roll_no"] = roll_match.group(1).strip() if roll_match else ""
    cert_match = re.search(r'(?:Certificate\s*(?:ID|No)|Cert\s*ID|ID[:\s]*)([A-Z0-9/]+)', text, re.IGNORECASE)
    extracted["cert_id"] = cert_match.group(1).strip() if cert_match else ""

    return extracted, ocr_conf, text

def normalize(text):
    return re.sub(r'[^\w]', '', text.lower().strip())

def fuzzy_lookup(name_norm, records_dict, threshold=85):
    best = process.extractOne(name_norm, records_dict.keys(), scorer=fuzz.ratio)
    if best and best[1] >= threshold:
        return best[0]
    return None

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="JACVS Bulk Verifier", layout="wide")
st.title("🛡️ JACVS - Bulk Certificate Verification")

st.sidebar.header("Instructions")
st.sidebar.write("- Upload a CSV: Columns `name, roll_no, cert_id`")
st.sidebar.write("- Upload one or multiple certificates (PDF/JPG/PNG)")
st.sidebar.write("- Ensure clear scans for best OCR accuracy")
debug = st.sidebar.checkbox("Enable Debug Mode")
st.session_state.debug = debug

# --- Upload CSV ---
csv_file = st.file_uploader("Upload student records CSV", type=['csv'])
records_dict = {}
if csv_file:
    try:
        df = pd.read_csv(csv_file)
        if all(col in df.columns for col in ['name','roll_no','cert_id']):
            records_dict = {normalize(row['name']): {"roll_no":normalize(row['roll_no']),
                                                    "cert_id":normalize(row['cert_id'])}
                            for _, row in df.iterrows()}
            st.success(f"Loaded {len(records_dict)} records from CSV.")
            if debug: st.write(records_dict)
        else:
            st.error("CSV must have columns: name, roll_no, cert_id")
    except Exception as e:
        st.error(f"CSV read error: {e}")

# --- Upload multiple certificates ---
uploaded_files = st.file_uploader("Upload certificate files (PDF/JPG/PNG)", type=['pdf','jpg','jpeg','png'], accept_multiple_files=True)

results = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        content = uploaded_file.read()
        uploaded_file.seek(0)
        st.subheader(f"📄 Processing: {uploaded_file.name}")

        # PDF handling
        if uploaded_file.type == "application/pdf":
            try:
                images = convert_from_bytes(content)
                if not images:
                    st.error("No pages found in PDF.")
                    continue
            except Exception as e:
                st.error(f"PDF error: {e}")
                continue
        else:
            images = [Image.open(io.BytesIO(content))]

        # OCR each page
        full_data = []
        for i,img in enumerate(images):
            extracted, conf, text = ocr_extract(img)
            full_data.append({"extracted":extracted, "conf":conf, "text":text})
            if debug: st.write(f"Page {i+1} OCR: {extracted}, Confidence: {conf:.1f}%")

        # Combine pages
        combined = {"name":"","roll_no":"","cert_id":"","full_text":""}
        for page in full_data:
            ext = page["extracted"]
            for k in ["name","roll_no","cert_id"]:
                if ext[k] and not combined[k]:
                    combined[k] = ext[k]
            combined["full_text"] += page["text"] + "\n"

        name_norm = normalize(combined["name"])
        roll_norm = normalize(combined["roll_no"])
        cert_norm = normalize(combined["cert_id"])

        # Document hash
        all_bytes = io.BytesIO()
        for img in images:
            temp = io.BytesIO()
            img.save(temp, format='PNG')
            all_bytes.write(temp.getvalue())
        doc_hash = hashlib.sha256(all_bytes.getvalue()).hexdigest()

        # Verification
        anomalies = []
        avg_conf = np.mean([p["conf"] for p in full_data])
        status = "Valid"
        rec = "Proceed with verification"
        confidence_score = int(avg_conf)

        if records_dict:
            matched_name = fuzzy_lookup(name_norm, records_dict)
            if matched_name:
                db_roll = records_dict[matched_name]["roll_no"]
                db_cert = records_dict[matched_name]["cert_id"]
                roll_score = fuzz.ratio(roll_norm, db_roll)
                cert_score = fuzz.ratio(cert_norm, db_cert)
                if roll_score < 85 or cert_score < 85:
                    anomalies.append("Mismatch in Roll No or Certificate ID")
                    status = "Caution"
                    confidence_score = min(confidence_score, 60)
                    rec = "Manual review recommended."
                else:
                    status = "Valid"
                    confidence_score = max(confidence_score, 90)
            else:
                anomalies.append("Name not found in CSV records")
                status = "Forged"
                confidence_score = min(confidence_score, 30)
                rec = "Document appears invalid."
        else:
            st.warning("No CSV uploaded. OCR only.")
            status = "OCR Processed"

        if avg_conf < 70:
            anomalies.append("Low OCR confidence")
            confidence_score = min(confidence_score, 30)
        if not anomalies and avg_conf>80:
            confidence_score=95

        result = {
            "file_name": uploaded_file.name,
            "status":status,
            "confidence_score":confidence_score,
            "recommendation":rec,
            "anomalies":anomalies,
            "extracted_data":combined,
            "document_hash":doc_hash,
            "avg_ocr_confidence":avg_conf
        }
        results.append(result)

        # Display per certificate
        col1,col2 = st.columns(2)
        with col1:
            st.markdown(f"**Status:** {'🟢' if status=='Valid' else '🟡' if status=='Caution' else '🔴'} {status} ({confidence_score}% Confidence)")
            st.write("Recommendation:", rec)
            if anomalies:
                st.error("⚠️ Anomalies:")
                for a in anomalies: st.write(f"- {a}")
            else:
                st.success("✅ No issues detected")

        with col2:
            st.subheader("Extracted Data")
            for k,v in combined.items():
                if k!="full_text" and v: st.write(f"**{k.title()}:** {v}")
            st.write(f"**Document Hash:** {doc_hash[:16]}...")
            if debug: st.text(combined["full_text"][:200])

# Download JSON report for all certificates
if results:
    report_json = json.dumps(results, indent=2)
    st.download_button("📥 Download Bulk Verification Report (JSON)", report_json, file_name="jacvs_bulk_report.json", mime="application/json")
