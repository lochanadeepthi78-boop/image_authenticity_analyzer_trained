import os
import uuid
import time
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from analyzer import analyze_image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# Extension check is case-insensitive by construction (everything is lower-cased
# before comparison). This is a first, cheap filter only - actual file content
# is verified separately below with Pillow so a mislabeled or corrupt file
# can't slip through, and a validly-named JPEG/JPG is never rejected because
# of it.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_extension(filename):
    """Robust, case-insensitive extension extraction. Returns '' if none."""
    _, dot, ext = filename.rpartition('.')
    return ext.lower() if dot else ''


def allowed_file(filename):
    return get_extension(filename) in ALLOWED_EXTENSIONS


def verify_image_content(file_storage):
    """
    Confirms the uploaded bytes are actually a readable image (not just that
    the filename ends in .jpg). Returns True/False and resets the stream
    position afterwards so the caller can still read/save the file.
    """
    try:
        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream)
        img.verify()  # cheap structural check, does not decode full pixel data
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False
    finally:
        file_storage.stream.seek(0)


def cleanup_old_uploads(max_age_hours=1):
    cutoff = time.time() - max_age_hours * 3600
    try:
        for f in os.listdir(UPLOAD_FOLDER):
            path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
    except Exception:
        pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    cleanup_old_uploads()

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use PNG, JPG, JPEG, BMP, WEBP, TIFF or GIF.'}), 400

    if not verify_image_content(file):
        return jsonify({'error': 'This file could not be read as a valid image. It may be corrupted or not really an image file.'}), 400

    ext = get_extension(file.filename) or 'jpg'
    unique_name = uuid.uuid4().hex + '.' + ext
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

    try:
        file.stream.seek(0)
        file.save(save_path)
    except Exception:
        return jsonify({'error': 'Could not save the uploaded file. Please try again.'}), 500

    file.stream.seek(0)

    try:
        result = analyze_image(file)
    except Exception as e:
        return jsonify({'error': 'Analysis failed: {}'.format(str(e))}), 500

    if 'error' in result:
        return jsonify(result), 400

    result['image_url'] = url_for('static', filename='uploads/' + unique_name)
    result['filename'] = os.path.basename(unique_name)

    return jsonify(result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('=' * 60)
    print('  Image Authenticity Analyzer')
    print('  Running at: http://127.0.0.1:{}'.format(port))
    print('  Press CTRL+C to stop.')
    print('=' * 60)
    app.run(debug=True, host='127.0.0.1', port=port)
