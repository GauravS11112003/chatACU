# Installing Tesseract OCR for Scanned Document Support

The app now supports **scanned PDFs** using OCR (Optical Character Recognition). You need to install Tesseract separately.

## Windows Installation

### 1. Download Tesseract

Get the latest installer from:
https://github.com/UB-Mannheim/tesseract/wiki

**Direct link (recommended):**
https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe

### 2. Install

- Run the installer (default location: `C:\Program Files\Tesseract-OCR\`)
- **IMPORTANT:** During installation, select **"Additional language data"** if you need languages other than English

### 3. Add to PATH (Option A - Automatic)

The installer should add Tesseract to your PATH automatically. Verify by opening a new PowerShell:

```powershell
tesseract --version
```

If this works, you're done!

### 4. Add to PATH (Option B - Manual)

If the command above fails:

1. Press `Win + X` → System → Advanced system settings → Environment Variables
2. Under "System variables", find `Path` and click Edit
3. Click New and add: `C:\Program Files\Tesseract-OCR`
4. Click OK on all windows
5. **Restart PowerShell** and test again: `tesseract --version`

### 5. Install Python Dependencies

```powershell
cd c:\Projects\chatACU
pip install -r requirements.txt
```

This installs:
- `pytesseract` - Python wrapper for Tesseract
- `pdf2image` - Converts PDF pages to images
- `pymupdf` (fitz) - Advanced PDF handling
- `pillow` - Image processing

### 6. Restart the App

```powershell
python -m streamlit run app.py
```

## Verify OCR Works

Upload a **scanned PDF** (image-based, not text). The app will:
1. Try normal text extraction first (fast)
2. If the page has no text, automatically use OCR
3. Show `📷 Page X: Used OCR (scanned document)` for scanned pages

## Common Issues

**"tesseract is not installed"**
- Tesseract not on PATH → add manually (see step 4)
- Restart your terminal after PATH changes

**"Failed to load tesseract"**
- Wrong PATH → ensure it points to `Tesseract-OCR` folder (not the exe)

**OCR quality is poor**
- Scan quality matters: 300 DPI recommended
- The app uses 2x upscaling automatically for better results
- Ensure PDFs are not heavily compressed

**Slow OCR**
- OCR takes ~1-3 seconds per page (vs <0.1s for text extraction)
- This is normal; scanned documents are much slower to process
- Progress bar shows current page being processed

## Languages

By default, Tesseract includes English. To add more languages:

1. During installation, select additional language data
2. Or download language files from: https://github.com/tesseract-ocr/tessdata
3. Place `.traineddata` files in: `C:\Program Files\Tesseract-OCR\tessdata\`

Then modify the OCR call in `app.py`:
```python
text = pytesseract.image_to_string(img, lang='eng+fra')  # English + French
```

## How It Works

The app now uses a **hybrid approach**:

1. **Text-based PDF pages** → Fast extraction with PyMuPDF (~0.1s/page)
2. **Scanned/image pages** → OCR with Tesseract (~2s/page)
3. Automatic detection: if a page has <50 characters, it's treated as scanned

This gives you the best of both worlds: speed for normal PDFs, OCR for scanned docs.
