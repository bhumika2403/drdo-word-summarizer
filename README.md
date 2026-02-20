# DRDO Word Summarizer System
**Offline Document Intelligence Terminal**

---

## Overview
A fully offline web application for searching keywords within uploaded documents (PDF, DOCX, TXT) and extracting contextual descriptions of the searched word directly from the document content.

---

## Features
- Upload PDF, DOCX, or TXT documents (up to 50 MB)
- Search any keyword within the document
- If word NOT found → clear notification displayed
- If word IS found → contextual description extracted from document, with:
  - Primary context (best/definition-like sentence)
  - Supporting context sentences
  - Word frequency count
  - Total context count
- 100% offline — no internet connection required
- DRDO-branded UI

---

## System Requirements
- Python 3.8 or above
- pip (Python package manager)

---

## Installation (One-Time Setup)

Open a terminal/command prompt in this folder and run:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:5000
```

> The application runs locally on your machine. No data is sent over the network.

---

## Usage
1. Click **Upload** or drag and drop a file (PDF/DOCX/TXT)
2. Wait for document to be processed (word count shown)
3. Type a keyword in the search box and press **Search** or hit Enter
4. View the extracted contextual description in the result panel

---

## Notes for Offline Deployment
- The UI uses Google Fonts (Rajdhani, Source Serif 4, JetBrains Mono) via a CDN link.
- For **fully offline** deployment with no internet on the machine:
  1. Download the font files from Google Fonts
  2. Place them in `static/fonts/`
  3. Update the `<link>` tag in `templates/index.html` to reference local font files using `@font-face`
  
  The app will still function normally without fonts — system fallback fonts will be used.

---

## File Structure
```
drdo_word_summarizer/
├── app.py              # Flask backend (core logic)
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── templates/
│   └── index.html      # Frontend UI (DRDO-themed)
└── uploads/            # Temporary file storage (auto-created)
```

---

*DRDO · Defence Research & Development Organisation · Ministry of Defence, Govt. of India*
