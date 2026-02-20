import os
import re
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pdfplumber
import docx

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath, ext):
    text = ""
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    elif ext == 'pdf':
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif ext == 'docx':
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text


def split_into_sentences(text):
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Also split on newlines for structured docs
    result = []
    for s in sentences:
        parts = [p.strip() for p in s.split('\n') if p.strip()]
        result.extend(parts)
    return [s for s in result if len(s) > 5]


def find_word_context(text, word):
    word_lower = word.lower().strip()
    sentences = split_into_sentences(text)

    # Find sentences containing the word (whole word match)
    pattern = re.compile(r'\b' + re.escape(word_lower) + r'\b', re.IGNORECASE)
    matched_sentences = [s for s in sentences if pattern.search(s)]

    # Count occurrences
    all_occurrences = pattern.findall(text)
    frequency = len(all_occurrences)

    if frequency == 0:
        return None, 0

    # Build a smart description:
    # 1. Pick the best sentence (longest one with the word in context)
    # 2. Include up to 5 surrounding sentences as context
    description_sentences = matched_sentences[:6]  # first 6 occurrences max

    # Try to find a "definition-like" sentence (contains: is, are, means, refers, defined)
    definition_patterns = re.compile(
        r'\b(is|are|means|refers to|defined as|describes|denotes|stands for|represents)\b',
        re.IGNORECASE
    )
    definition_sentences = [s for s in matched_sentences if definition_patterns.search(s)]

    primary = definition_sentences[0] if definition_sentences else matched_sentences[0]
    supporting = [s for s in matched_sentences if s != primary][:4]

    return {
        "word": word,
        "frequency": frequency,
        "primary_context": primary,
        "supporting_contexts": supporting,
        "total_sentences_found": len(matched_sentences),
    }, frequency


def summarize_text(text, num_sentences=8):
    """Extractive summarization using sentence scoring (fully offline, no ML needed)."""
    sentences = split_into_sentences(text)
    if not sentences:
        return [], {}

    # Step 1: Build word frequency table (ignore stopwords)
    stopwords = set([
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'is','are','was','were','be','been','being','have','has','had','do','does',
        'did','will','would','could','should','may','might','shall','can','this',
        'that','these','those','it','its','by','from','as','into','through','during',
        'before','after','above','below','between','each','so','such','than','too',
        'very','just','also','about','up','out','if','then','there','when','where',
        'which','who','whom','how','all','both','few','more','most','other','some',
        'any','only','same','own','not','no','nor','he','she','they','we','you','i',
    ])

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    for w in words:
        if w not in stopwords:
            word_freq[w] = word_freq.get(w, 0) + 1

    # Normalize frequencies
    max_freq = max(word_freq.values()) if word_freq else 1
    for w in word_freq:
        word_freq[w] /= max_freq

    # Step 2: Score each sentence
    sentence_scores = {}
    for sent in sentences:
        sent_words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
        score = sum(word_freq.get(w, 0) for w in sent_words)
        # Prefer medium-length sentences (not too short, not too long)
        length = len(sent_words)
        if length < 5:
            score *= 0.5
        elif length > 40:
            score *= 0.7
        sentence_scores[sent] = score

    # Step 3: Pick top N sentences in original order
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    # Maintain original document order
    ordered = [s for s in sentences if s in top_sentences]

    # Step 4: Key topics = top 10 most frequent non-stopword words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    key_topics = [w for w, _ in sorted_words[:12] if len(w) > 3]

    return ordered, key_topics


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Use PDF, DOCX, or TXT.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    ext = filename.rsplit('.', 1)[1].lower()
    try:
        text = extract_text_from_file(filepath, ext)
    except Exception as e:
        return jsonify({'error': f'Failed to parse file: {str(e)}'}), 500

    if not text.strip():
        return jsonify({'error': 'File appears to be empty or unreadable.'}), 400

    # Count basic stats
    word_count = len(text.split())
    sentence_count = len(split_into_sentences(text))

    # Store text temporarily (use session-safe approach via hidden storage)
    text_file = filepath + '.extracted.txt'
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text)

    return jsonify({
        'success': True,
        'filename': filename,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'text_key': filename,  # used to retrieve text for search
    })


@app.route('/search', methods=['POST'])
def search_word():
    data = request.get_json()
    word = data.get('word', '').strip()
    filename = data.get('filename', '').strip()

    if not word:
        return jsonify({'error': 'Please enter a word to search.'}), 400
    if not filename:
        return jsonify({'error': 'No document loaded.'}), 400

    safe_filename = secure_filename(filename)
    text_file = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename + '.extracted.txt')

    if not os.path.exists(text_file):
        return jsonify({'error': 'Document session expired. Please re-upload the file.'}), 404

    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    result, frequency = find_word_context(text, word)

    if frequency == 0:
        return jsonify({
            'found': False,
            'word': word,
            'message': f'The word "{word}" was not found in the document.'
        })

    return jsonify({
        'found': True,
        **result
    })


@app.route('/visualize', methods=['POST'])
def visualize_document():
    try:
        data = request.get_json()
        filename = data.get('filename', '').strip()
        search_word = data.get('word', '').strip()

        if not filename:
            return jsonify({'error': 'No document loaded.'}), 400

        safe_filename = secure_filename(filename)
        text_file = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename + '.extracted.txt')

        if not os.path.exists(text_file):
            return jsonify({'error': 'Document session expired. Please re-upload the file.'}), 404

        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        stopwords = set([
            'the','a','an','and','or','but','in','on','at','to','for','of','with',
            'is','are','was','were','be','been','being','have','has','had','do','does',
            'did','will','would','could','should','may','might','shall','can','this',
            'that','these','those','it','its','by','from','as','into','through','during',
            'before','after','above','below','between','each','so','such','than','too',
            'very','just','also','about','up','out','if','then','there','when','where',
            'which','who','whom','how','all','both','few','more','most','other','some',
            'any','only','same','own','not','no','nor','he','she','they','we','you','i',
            'its','our','their','your','his','her','my','am','us','me','him','her',
        ])

        # Top 12 keyword frequencies
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for w in words:
            if w not in stopwords:
                word_freq[w] = word_freq.get(w, 0) + 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:12]

        # Word position distribution
        word_positions = []
        if search_word:
            pattern = re.compile(r'\b' + re.escape(search_word) + r'\b', re.IGNORECASE)
            total_chars = len(text)
            for m in pattern.finditer(text):
                pct = round((m.start() / total_chars) * 100, 1)
                word_positions.append(pct)

        # Sentence length distribution
        sentences = split_into_sentences(text)
        length_buckets = {'1-10': 0, '11-20': 0, '21-30': 0, '31-50': 0, '50+': 0}
        for s in sentences:
            wc = len(s.split())
            if wc <= 10:     length_buckets['1-10'] += 1
            elif wc <= 20:   length_buckets['11-20'] += 1
            elif wc <= 30:   length_buckets['21-30'] += 1
            elif wc <= 50:   length_buckets['31-50'] += 1
            else:            length_buckets['50+'] += 1

        return jsonify({
            'top_words': [{'word': w, 'count': c} for w, c in top_words],
            'word_positions': word_positions,
            'sentence_lengths': length_buckets,
            'total_words': len(words),
            'total_sentences': len(sentences),
            'search_word': search_word,
            'search_count': len(word_positions),
        })
    except Exception as e:
        return jsonify({'error': f'Chart generation failed: {str(e)}'}), 500


@app.route('/summarize', methods=['POST'])
def summarize_document():
    data = request.get_json()
    filename = data.get('filename', '').strip()

    if not filename:
        return jsonify({'error': 'No document loaded.'}), 400

    safe_filename = secure_filename(filename)
    text_file = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename + '.extracted.txt')

    if not os.path.exists(text_file):
        return jsonify({'error': 'Document session expired. Please re-upload the file.'}), 404

    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    summary_sentences, key_topics = summarize_text(text, num_sentences=8)
    total_sentences = len(split_into_sentences(text))
    word_count = len(text.split())

    return jsonify({
        'success': True,
        'summary': summary_sentences,
        'key_topics': key_topics,
        'total_sentences': total_sentences,
        'word_count': word_count,
        'compression': f"{len(summary_sentences)}/{total_sentences} sentences"
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
