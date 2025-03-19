# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pytesseract
# from PIL import Image
# import mysql.connector
# import re
# from datetime import datetime
# import io
# from pdf2image import convert_from_bytes
# import cv2
# import numpy as np
# import os

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# # Ensure "debug_images" folder exists (for saving debug images)
# os.makedirs("debug_images", exist_ok=True)

# def connect_db():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="Broseph18",
#         database="invoices_db"
#     )

# def parse_date(raw_date):
#     possible_formats = ["%d-%m-%y", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"]
#     for fmt in possible_formats:
#         try:
#             return datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
#         except ValueError:
#             pass
#     return None

# def format_amount(amount_str):
#     try:
#         return float(amount_str.replace(",", "").replace("$", ""))
#     except ValueError:
#         return None

# def deskew_image(img_cv):
#     """
#     Deskew only if angle > 10 degrees to avoid over-rotating slightly skewed images.
#     """
#     gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bitwise_not(gray)
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#     coords = np.column_stack(np.where(thresh > 0))
#     angle = 0.0
#     if len(coords) > 0:
#         rect = cv2.minAreaRect(coords)
#         angle = rect[-1]
#         if angle < -45:
#             angle = -(90 + angle)
#         else:
#             angle = -angle

#     if abs(angle) > 10:
#         (h, w) = img_cv.shape[:2]
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#         return rotated
#     else:
#         return img_cv

# def upscale_pil_image(pil_image, scale=2):
#     """
#     Upscale the PIL image by a given factor (2x by default) to improve OCR on small text.
#     """
#     new_width = pil_image.width * scale
#     new_height = pil_image.height * scale
#     return pil_image.resize((new_width, new_height), Image.LANCZOS)

# def preprocess_image(pil_image, doc_style, debug_prefix=""):
#     # (Optional) Upscale if text is really small. Otherwise, skip or reduce scale=1.5
#     pil_image = upscale_pil_image(pil_image, scale=2)

#     img_cv = np.array(pil_image)
#     img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

#     # Deskew
#     img_cv = deskew_image(img_cv)

#     gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

#     # Contrast Enhancement
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     gray = clahe.apply(gray)

#     # Try a bigger block size for adaptive threshold
#     block_size = 31  # or 41
#     C = 8
#     # thresh = cv2.adaptiveThreshold(
#     #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#     #     cv2.THRESH_BINARY, block_size, C
#     # )

#     # If you still get stripes, try Otsu instead:
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     if doc_style.lower() == "handwritten":
#         thresh = cv2.medianBlur(thresh, 3)

#     # Light morphological close to remove lines
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#     debug_path = f"debug_images/{debug_prefix}_preprocessed.png"
#     cv2.imwrite(debug_path, thresh)

#     processed_pil = Image.fromarray(thresh)
#     return processed_pil


# @app.route("/upload-invoice", methods=["POST"])
# def upload_invoice():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     doc_type = request.form.get("doc_type", "invoice").lower()
#     doc_style = request.form.get("doc_style", "digital").lower()

#     # Tesseract config (Try psm 4 for multi-column, psm 11 or 6 for single/sparse text)
#     if doc_style == "handwritten":
#         OCR_CONFIG = "--oem 1 --psm 11"
#     else:
#         OCR_CONFIG = "--oem 3 --psm 4"

#     file = request.files["file"]
#     filename = file.filename.lower()

#     text = ""
#     try:
#         if filename.endswith(".pdf"):
#             pages = convert_from_bytes(file.read())
#             for i, page in enumerate(pages, start=1):
#                 # Preprocess each PDF page
#                 preprocessed = preprocess_image(page, doc_style, debug_prefix=f"pdf_page_{i}")
#                 # OCR
#                 page_text = pytesseract.image_to_string(preprocessed, config=OCR_CONFIG)
#                 text += page_text + "\n"
#         else:
#             pil_image = Image.open(io.BytesIO(file.read()))
#             preprocessed = preprocess_image(pil_image, doc_style, debug_prefix="image")
#             text = pytesseract.image_to_string(preprocessed, config=OCR_CONFIG)
#     except Exception as e:
#         return jsonify({"error": f"Error processing file: {str(e)}"}), 500

#     print("Final OCR Output:\n", text)  # Debugging OCR output

#     # If you only care about the raw OCR text for now, just return it:
#     return jsonify({"status": "success", "raw_ocr_text": text})

# @app.route("/confirm-invoice", methods=["POST"])
# def confirm_invoice():
#     data = request.json

#     # Save to DB if needed, or skip if you're just debugging OCR
#     conn = connect_db()
#     cursor = conn.cursor()
#     insert_query = """
#     INSERT INTO invoice_data (invoice_number, invoice_date, invoice_amount)
#     VALUES (%s, %s, %s)
#     """
#     cursor.execute(
#         insert_query,
#         (data["invoice_number"], data["invoice_date"], data["invoice_amount"])
#     )
#     conn.commit()
#     conn.close()

#     return jsonify({"status": "success", "message": "Invoice data saved successfully"})

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from PIL import Image
import mysql.connector
import re
from datetime import datetime
import io
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import os
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("invoice_processor.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Create folders for debugging
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

def connect_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="Broseph18",
            database="invoices_db"
        )
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        raise

def parse_date(raw_date):
    possible_formats = ["%d-%m-%y", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"]
    for fmt in possible_formats:
        try:
            return datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None

def format_amount(amount_str):
    if not amount_str:
        return None
    # More robust amount extraction
    # Remove all non-digit characters except for decimal point and negative sign
    clean_str = re.sub(r'[^\d.-]', '', amount_str)
    try:
        return float(clean_str)
    except ValueError:
        return None

def deskew_image(img_cv):
    """
    Deskew the image if it's tilted beyond a threshold.
    """
    try:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = 0.0
        if len(coords) > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

        # Only deskew if angle is significant
        if abs(angle) > 5:
            logger.info(f"Deskewing image with angle: {angle}")
            (h, w) = img_cv.shape[:2]
            center = (w // 2, h // 2)
            
            # Calculate new dimensions to avoid cropping
            # Convert angle to radians
            angle_rad = np.radians(angle)
            
            # Calculate new width and height
            new_w = int(abs(h * np.sin(angle_rad)) + abs(w * np.cos(angle_rad)))
            new_h = int(abs(h * np.cos(angle_rad)) + abs(w * np.sin(angle_rad)))
            
            # Adjust rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Update translation component of the matrix
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            
            # Apply rotation with new dimensions
            rotated = cv2.warpAffine(img_cv, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return img_cv
    except Exception as e:
        logger.error(f"Error in deskew_image: {str(e)}")
        return img_cv  # Return original image if deskewing fails
def upscale_pil_image(pil_image, scale=1.5):
    """
    Upscale the PIL image to improve OCR on small text.
    Reduced default scale from 2 to 1.5 to prevent over-enlargement.
    """
    try:
        new_width = int(pil_image.width * scale)
        new_height = int(pil_image.height * scale)
        return pil_image.resize((new_width, new_height), Image.LANCZOS)
    except Exception as e:
        logger.error(f"Error in upscale_pil_image: {str(e)}")
        return pil_image  # Return original image if resizing fails

def preprocess_image(pil_image, doc_style, debug_prefix=""):
    """
    Preprocess image with multiple approaches and save debug images for each step.
    """
    session_id = f"{debug_prefix}_{uuid.uuid4().hex[:8]}"
    
    try:
        # Save original image for debugging
        original_debug_path = f"{DEBUG_DIR}/{session_id}_1_original.png"
        # pil_image.save(original_debug_path)
        logger.info(f"Saved original image to {original_debug_path}")

        # Less aggressive scaling (1.5x instead of 2x)
        pil_image = upscale_pil_image(pil_image, scale=1.5)
        upscaled_debug_path = f"{DEBUG_DIR}/{session_id}_2_upscaled.png"
        # pil_image.save(upscaled_debug_path)

        # Convert to OpenCV format
        img_cv = np.array(pil_image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Deskew
        img_cv = deskew_image(img_cv)
        cv2.imwrite(f"{DEBUG_DIR}/{session_id}_3_deskewed.png", img_cv)

        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{DEBUG_DIR}/{session_id}_4_grayscale.png", gray)

        # Try multiple preprocessing approaches and save each
        
        # Approach 1: CLAHE for contrast enhancement (less aggressive)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Reduced from 3.0
        clahe_img = clahe.apply(gray)
        cv2.imwrite(f"{DEBUG_DIR}/{session_id}_5_clahe.png", clahe_img)
        
        # Approach 2: Otsu thresholding
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(f"{DEBUG_DIR}/{session_id}_6_otsu.png", otsu_thresh)
        
        # Approach 3: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 25, 10  # Adjusted parameters
        )
        cv2.imwrite(f"{DEBUG_DIR}/{session_id}_7_adaptive.png", adaptive_thresh)
        
        # Choose the best preprocessing method based on document style
        if doc_style.lower() == "handwritten":
            final_img = cv2.medianBlur(otsu_thresh, 3)
        else:
            # For digital documents, adaptive thresholding often works better
            final_img = adaptive_thresh
        
        # Light noise removal
        if doc_style.lower() != "handwritten":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            final_img = cv2.morphologyEx(final_img, cv2.MORPH_CLOSE, kernel)
            final_img = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel)
            
        # Save final processed image
        final_debug_path = f"{DEBUG_DIR}/{session_id}_8_final.png"
        cv2.imwrite(final_debug_path, final_img)
        logger.info(f"Saved final preprocessed image to {final_debug_path}")
        
        processed_pil = Image.fromarray(final_img)
        return processed_pil, session_id
    
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        logger.error(f"Returning original image due to preprocessing failure")
        return pil_image, session_id  # Return original image if processing fails

def extract_invoice_details(text, doc_type="invoice"):
    """
    Extract key information from OCR text based on document type.
    """
    result = {
        "raw_text": text,
        "invoice_number": None,
        "invoice_date": None,
        "invoice_amount": None,
        "extracted_fields": {}
    }
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Extract invoice number
    invoice_number_patterns = [
        r"invoice\s*(?:#|number|num|no|num\.:|no\.:|number:)\s*([A-Za-z0-9-_/]+)",
        r"inv\s*(?:#|number|num|no|num\.:|no\.:|number:)\s*([A-Za-z0-9-_/]+)",
        r"invoice\s*id\s*(?::|#)?\s*([A-Za-z0-9-_/]+)",
        r"bill\s*(?:#|number|num|no|num\.:|no\.:|number:)\s*([A-Za-z0-9-_/]+)"
    ]
    
    for pattern in invoice_number_patterns:
        matches = re.search(pattern, text_lower)
        if matches:
            result["invoice_number"] = matches.group(1).strip()
            break
    
    # Extract date
    date_patterns = [
        r"(?:invoice|bill|order|statement|due)?\s*date\s*(?::|is)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?:invoice|bill|order|statement|due)?\s*date\s*(?::|is)?\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2})",
        r"dated?\s*(?::|is)?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"dated?\s*(?::|is)?\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2})"
    ]
    
    for pattern in date_patterns:
        matches = re.search(pattern, text_lower)
        if matches:
            raw_date = matches.group(1).strip()
            parsed_date = parse_date(raw_date)
            if parsed_date:
                result["invoice_date"] = parsed_date
                break
    
    # Extract amount
    amount_patterns = [
        r"(?:total|amount|sum|invoice amount|invoice total)\s*(?:due|:)?\s*(?:[$£€])\s*([0-9,.]+)",
        r"(?:total|amount|sum|invoice amount|invoice total)\s*(?:due|:)?\s*([0-9,.]+)\s*(?:[$£€])",
        r"(?:[$£€])\s*([0-9,.]+)\s*(?:total|amount|sum|invoice amount|invoice total)"
    ]
    
    for pattern in amount_patterns:
        matches = re.search(pattern, text_lower)
        if matches:
            amount_str = matches.group(1).strip()
            amount = format_amount(amount_str)
            if amount is not None:
                result["invoice_amount"] = amount
                break
    
    return result

@app.route("/upload-invoice", methods=["POST"])
def upload_invoice():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    doc_type = request.form.get("doc_type", "invoice").lower()
    doc_style = request.form.get("doc_style", "digital").lower()
    
    logger.info(f"Processing {doc_type} with style {doc_style}")

    # Configure Tesseract based on document style
    if doc_style == "handwritten":
        # For handwritten text
        OCR_CONFIG = "--oem 1 --psm 6 -l eng --dpi 300"
    else:
        # For printed/digital text - try different PSM modes depending on layout
        OCR_CONFIG = "--oem 3 --psm 3 -l eng --dpi 300"
    
    file = request.files["file"]
    filename = file.filename.lower() if file.filename else "unknown_file"
    
    logger.info(f"Processing file: {filename}")
    
    text = ""
    debug_ids = []
    
    try:
        if filename.endswith(".pdf"):
            # Process PDF
            logger.info("Processing PDF file")
            pdf_bytes = file.read()
            
            try:
                pages = convert_from_bytes(pdf_bytes, dpi=300)
                logger.info(f"PDF converted successfully with {len(pages)} pages")
                
                for i, page in enumerate(pages, start=1):
                    logger.info(f"Processing page {i} of PDF")
                    
                    # Preprocess each PDF page
                    preprocessed, debug_id = preprocess_image(page, doc_style, debug_prefix=f"pdf_page_{i}")
                    debug_ids.append(debug_id)
                    
                    # Try OCR with different configs if needed
                    page_text = pytesseract.image_to_string(preprocessed, config=OCR_CONFIG)
                    
                    if not page_text.strip():
                        logger.warning(f"No text detected on page {i} with primary config. Trying alternative.")
                        # Try alternative config
                        alt_config = "--oem 3 --psm 6 -l eng --dpi 300"
                        page_text = pytesseract.image_to_string(preprocessed, config=alt_config)
                    
                    text += page_text + "\n"
                    logger.info(f"OCR completed for page {i}")
            
            except Exception as e:
                logger.error(f"Error converting PDF: {str(e)}")
                return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500
                
        else:
            # Process image file
            logger.info("Processing image file")
            try:
                pil_image = Image.open(io.BytesIO(file.read()))
                preprocessed, debug_id = preprocess_image(pil_image, doc_style, debug_prefix="image")
                debug_ids.append(debug_id)
                
                # Try OCR with primary config
                text = pytesseract.image_to_string(preprocessed, config=OCR_CONFIG)
                
                # If no text detected, try alternative configuration
                if not text.strip():
                    logger.warning("No text detected with primary config. Trying alternative.")
                    # Try different PSM modes
                    alt_configs = ["--oem 3 --psm 6 -l eng --dpi 300", "--oem 3 --psm 4 -l eng --dpi 300"]
                    
                    for alt_config in alt_configs:
                        alt_text = pytesseract.image_to_string(preprocessed, config=alt_config)
                        if alt_text.strip():
                            text = alt_text
                            logger.info("Alternative OCR config successful")
                            break
            
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error during file processing: {str(e)}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

    # Log OCR output for debugging
    # Log OCR output for debugging (limited for logs)
    logger.info(f"OCR Output (first 500 chars): {text[:500]}...")

    # Write the full OCR text to a separate file for complete debugging
    with open(f"{DEBUG_DIR}/_ocr_full.txt", "w", encoding="utf-8") as f:
        f.write(text)

    # Extract invoice details
    extracted_data = extract_invoice_details(text, doc_type)

    # Return the FULL text in the response
    response = {
        "status": "success", 
        "raw_ocr_text": text,  # Full text here, not truncated
        "extracted_data": extracted_data,
        "debug_ids": debug_ids
    }
    
    # Extract invoice details
    extracted_data = extract_invoice_details(text, doc_type)
    
    # Return both raw text and extracted data
    response = {
        "status": "success", 
        "raw_ocr_text": text,
        "extracted_data": extracted_data,
        "debug_ids": debug_ids
    }
    
    return jsonify(response)

@app.route("/confirm-invoice", methods=["POST"])
def confirm_invoice():
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Log the received data
        logger.info(f"Received data for confirmation: {data}")
        
        # Validate required fields
        required_fields = ["invoice_number", "invoice_date", "invoice_amount"]
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        
        if missing_fields:
            return jsonify({
                "error": "Missing required fields", 
                "missing_fields": missing_fields
            }), 400
        
        # Connect to database
        try:
            conn = connect_db()
            cursor = conn.cursor()
            
            # Insert into database
            insert_query = """
            INSERT INTO invoice_data (invoice_number, invoice_date, invoice_amount)
            VALUES (%s, %s, %s)
            """
            
            cursor.execute(
                insert_query,
                (data["invoice_number"], data["invoice_date"], data["invoice_amount"])
            )
            
            conn.commit()
            logger.info(f"Invoice data saved successfully: {data['invoice_number']}")
            
            # Close connection
            cursor.close()
            conn.close()
            
            return jsonify({
                "status": "success", 
                "message": "Invoice data saved successfully",
                "invoice_id": data["invoice_number"]
            })
            
        except mysql.connector.Error as err:
            logger.error(f"Database error: {err}")
            return jsonify({"error": f"Database error: {str(err)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in confirm_invoice: {str(e)}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route("/debug-info", methods=["GET"])
def debug_info():
    """
    Endpoint to check system info for debugging purposes.
    """
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        return jsonify({
            "status": "success",
            "tesseract_version": str(tesseract_version),
            "debug_dir_exists": os.path.exists(DEBUG_DIR),
            "python_version": os.sys.version
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)