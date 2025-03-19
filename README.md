# Bill and Invoice OCR Processor

A Flask-based web application for extracting data from invoices using Optical Character Recognition (OCR). This system processes both PDF and image files, applies various preprocessing techniques to improve OCR accuracy, and extracts key invoice information such as invoice numbers, dates, and amounts.
It has been made by keeping the Invoices and Bills in mind but its not limited to them, the OCR recognises pretty much all the text available on the image and the RegEx (Regular Expression) extracts all the relevant text from the given file or Image.

Many debug images will be generated during testing to help you to better grasp what's happening underneath and help you further fine tune the model. Feel free to update the model and the RegEx to suite your needs as this is just limited to a use case and can be further improved by modofying the RegEx. It is a Basic Image to Text Model that you can tweak as per your own needs.

## Features

- **Free and Open Source**: Completely self-contained ML model that doesn't require any paid APIs or services
- **Comparable to Paid Solutions**: Achieves accuracy similar to commercial OCR services through advanced preprocessing
- **Document Processing**: Handles both PDF and image files (JPG, PNG, etc.)
- **Image Preprocessing**: Applies various image enhancement techniques to improve OCR accuracy
  - Deskewing (correcting tilted images)
  - Upscaling for better text recognition
  - Contrast enhancement
  - Multiple thresholding methods
- **Adaptive OCR**: Uses different Tesseract configurations based on document style (printed vs. handwritten)
- **Data Extraction**: Automatically identifies and extracts:
  - Invoice numbers
  - Invoice dates (with multiple format support)
  - Invoice amounts
- **Debug Capabilities**: Saves intermediate processing steps as images for debugging and tuning
- **Database Integration**: Stores extracted invoice data in MySQL database

## Technical Architecture

### Backend (Flask)

- RESTful API endpoints for document upload and data confirmation
- Image preprocessing using OpenCV
- OCR via Tesseract
- PDF processing using pdf2image
- MySQL database integration

### Preprocessing Pipeline

1. **Initial Processing**:

   - Upscaling to improve resolution of small text
   - Deskewing to correct tilted documents

2. **Enhancement**:

   - Grayscale conversion
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Otsu and adaptive thresholding

3. **Document-Type Specific Processing**:
   - Different approaches for handwritten vs. printed documents
   - Noise removal using morphological operations

## Setup and Installation

### Prerequisites

- Python 3.6+
- Tesseract OCR engine
- Poppler (for PDF processing)
- MySQL database

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/invoice-ocr-processor.git
   cd invoice-ocr-processor
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies**:

   - Tesseract OCR: [Installation instructions](https://github.com/tesseract-ocr/tesseract)
   - Poppler: [Installation instructions](https://poppler.freedesktop.org/)

5. **Set up MySQL database**:

   ```sql
   CREATE DATABASE invoices_db;
   USE invoices_db;
   CREATE TABLE invoice_data (
       id INT AUTO_INCREMENT PRIMARY KEY,
       invoice_number VARCHAR(255),
       invoice_date DATE,
       invoice_amount DECIMAL(10, 2),
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

6. **Update database connection details**:
   - Edit the `connect_db()` function in the application to match your MySQL credentials

## Usage

1. **Start the Flask server**:

   ```bash
   python app.py
   ```

2. **API Endpoints**:

   - `POST /upload-invoice`: Upload and process an invoice
     - Form parameters:
       - `file`: Invoice file (PDF/image)
       - `doc_type`: Document type (default: "invoice")
       - `doc_style`: Document style (default: "digital", options: "digital" or "handwritten")
   - `POST /confirm-invoice`: Save extracted invoice data to database
     - JSON body with invoice_number, invoice_date, and invoice_amount
   - `GET /debug-info`: Get system information for debugging

3. **Example API Request**:
   ```bash
   curl -X POST -F "file=@invoice.pdf" -F "doc_style=digital" \
        http://localhost:5000/upload-invoice
   ```

## Debugging

The application includes extensive debugging capabilities:

- Debug images are saved in the `debug_images` directory
- Processing steps are logged in `invoice_processor.log`
- Full OCR output is saved to separate files for analysis
- Debug endpoint provides system information

## Key Improvements

The recent update includes:

1. **Enhanced Deskewing Algorithm**: Fixed cropping issues by calculating proper dimensions for rotated images
2. **Complete OCR Output**: Fixed truncation of OCR text in responses and improved debugging output
3. **Multiple Preprocessing Approaches**: Different techniques for various document types
4. **Robust Error Handling**: Better logging and graceful failure modes

## Why Choose This Solution

- **Cost-effective**: No recurring API costs or licensing fees
- **Privacy-focused**: All processing happens locally with no data sent to third-party services
- **Customizable**: Full control over the preprocessing pipeline for specific document types
- **Accuracy**: Custom preprocessing chain delivers results comparable to commercial solutions
- **Open Source**: Benefit from community improvements and contribute back

## License

This project is licensed under the **MIT License**, allowing free use, modification, and distribution with attribution.

## Acknowledgements

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV](https://opencv.org/)
- [Flask](https://flask.palletsprojects.com/)
