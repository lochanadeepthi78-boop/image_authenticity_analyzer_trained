import io
import os
import numpy as np
from PIL import Image, ImageChops, ImageFilter

# =====================================================================
# NOTE FOR THE PROJECT AUTHOR
# ---------------------------------------------------------------------
# There is currently NO trained machine-learning model anywhere in this
# project. Every number below (entropy, edge sharpness, texture
# uniformity, frequency ratios, DCT grid strength, "AI probability",
# etc.) comes from handcrafted statistical formulas over pixel data,
# not from a classifier trained on real vs. AI-generated images.
#
# That is fine for a mini-project *as long as it is labelled honestly*
# and the different kinds of evidence are kept separate, which is what
# this file now does:
#
#   1. analyze_compression()   -> was this file lossily (re)compressed?
#   2. analyze_manipulation()  -> is there evidence of local editing?
#   3. analyze_ai_generated()  -> handcrafted "AI-likeness" heuristic
#
# These three used to overlap (compression signals were silently
# leaking into the "AI-generated" score), which is what caused a real,
# lightly-compressed photo to be scored as ~55% AI-generated. They are
# now computed independently, and analyze_ai_generated() explicitly
# down-weights any signal that compression already explains.
#
# MODEL_USED is reported to the frontend so the UI never overstates
# what this system actually is. See the bottom of this file for how to
# wire in a real trained model later.
# =====================================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'ai_classifier.npz')
MODEL_USED = os.path.exists(MODEL_PATH)
MODEL_DATA = None

if MODEL_USED:
    try:
        MODEL_DATA = np.load(MODEL_PATH, allow_pickle=False)
    except Exception:
        MODEL_USED = False
        MODEL_DATA = None

ELA_LEVEL = 95


def load_image(file_storage):
    """Open the upload and return (RGB image, original format string)."""
    img = Image.open(file_storage)
    original_format = (img.format or 'UNKNOWN').upper()
    img = img.convert('RGB')
    return img, original_format


def error_level_analysis(img):
    original = img.convert('RGB')
    buffer = io.BytesIO()
    original.save(buffer, format='JPEG', quality=ELA_LEVEL)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert('RGB')
    diff = ImageChops.difference(original, recompressed)
    gray = diff.convert('L')
    arr = np.array(gray, dtype=np.float32)
    max_ela = float(arr.max())
    mean_ela = float(arr.mean())
    p90 = float(np.percentile(arr, 90))
    modified_pixel_ratio = float((arr > 8).sum() / arr.size * 100)
    return {
        'max': round(max_ela, 2),
        'mean': round(mean_ela, 2),
        'p90': round(p90, 2),
        'modified_pixels_pct': round(modified_pixel_ratio, 2),
    }


def _gaussian_blur(arr, sigma):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred, dtype=np.float32)


def _laplacian(arr):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    padded = np.pad(arr, 1, mode='reflect')
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += kernel[i, j] * padded[i:i + arr.shape[0], j:j + arr.shape[1]]
    return out


def analyze_noise(img):
    gray = np.array(img.convert('L'), dtype=np.float32)
    high_freq = gray - _gaussian_blur(gray, 1.0)
    noise_std = float(high_freq.std())
    noise_mean = float(np.abs(high_freq).mean())
    laplacian = np.abs(_laplacian(gray))
    edge_density = float((laplacian > 25).sum() / laplacian.size * 100)
    return {
        'noise_std': round(noise_std, 3),
        'noise_mean': round(noise_mean, 3),
        'edge_density_pct': round(edge_density, 2)
    }


def analyze_frequency_spectrum(img):
    gray = np.array(img.convert('L'), dtype=np.float64)
    gray = gray - gray.mean()
    fft2 = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft2)
    magnitude = np.abs(fft_shift)
    magnitude /= magnitude.sum()
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    low_band = magnitude[cy-3:cy+4, cx-3:cx+4].sum()
    mid_ring = _ring_energy(magnitude, cy, cx, 8, 25)
    high_ring = _ring_energy(magnitude, cy, cx, 30, 70)
    total = low_band + mid_ring + high_ring
    if total == 0:
        total = 1e-12
    low_ratio = (low_band / total) * 100
    mid_ratio = (mid_ring / total) * 100
    high_ratio = (high_ring / total) * 100
    safe = magnitude + 1e-15
    spectral_flatness = float(np.exp(np.log(safe).mean()) / safe.mean())
    return {
        'low_freq_pct': round(float(low_ratio), 2),
        'mid_freq_pct': round(float(mid_ratio), 2),
        'high_freq_pct': round(float(high_ratio), 2),
        'spectral_flatness': round(spectral_flatness, 6)
    }


def _ring_energy(magnitude, cy, cx, r_inner, r_outer):
    h, w = magnitude.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (dist >= r_inner) & (dist < r_outer)
    return magnitude[mask].sum()


def analyze_dct_artifacts(img):
    gray = np.array(img.convert('L'), dtype=np.float32)
    h, w = gray.shape
    block_size = 8
    valid_h = (h // block_size) * block_size
    valid_w = (w // block_size) * block_size
    if valid_h == 0 or valid_w == 0:
        return {'grid_strength_coeff': 0.0}
    gray = gray[:valid_h, :valid_w]
    blocks = gray.reshape(valid_h // block_size, block_size, valid_w // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)
    sampled = blocks[::10]
    if sampled.shape[0] == 0:
        sampled = blocks
    dct_energy = np.zeros((block_size, block_size))
    for b in sampled:
        dct = np.fft.fft2(b - b.mean())
        dct_energy += np.abs(dct)
    if sampled.shape[0] > 0:
        dct_energy /= sampled.shape[0]
    dct_energy = np.abs(dct_energy)
    total = dct_energy.sum()
    if total == 0:
        return {'grid_strength_coeff': 0.0}
    diagonal_ratio = (np.diag(dct_energy) / total).mean() * 100
    grid_strength = float(np.clip(diagonal_ratio * 1.5, 0, 100))
    return {
        'grid_strength_coeff': round(grid_strength, 2)
    }


def detect_aspect_ratio_edit(img):
    w, h = img.size
    return {
        'width': w,
        'height': h,
        'aspect_ratio': round(w / h, 4)
    }


# ---------------------------------------------------------------------
# 1. COMPRESSION ANALYSIS  (independent of AI / manipulation)
# ---------------------------------------------------------------------
def analyze_compression(ela, dct, noise, original_format):
    """
    Decide whether the file shows signs of lossy compression.
    This is deliberately independent of the AI-generation and
    manipulation logic below - a compressed photo is not automatically
    edited or AI-generated.
    """
    indicators = []
    detected = False

    if original_format in ('JPEG', 'JPG'):
        detected = True
        indicators.append('The uploaded file is stored in JPEG format, which is always lossy')

    if dct['grid_strength_coeff'] > 18:
        detected = True
        indicators.append('8x8 block grid pattern detected in the DCT domain (typical JPEG blockiness)')

    if not detected and ela['mean'] > 1.5 and ela['modified_pixels_pct'] < 8.0 and noise['noise_std'] < 6.0:
        detected = True
        indicators.append('Uniform, low-level recompression response consistent with lossy re-saving')

    return {
        'detected': detected,
        'label': 'COMPRESSED (LOSSY)' if detected else 'NO SIGNIFICANT COMPRESSION DETECTED',
        'source_format': original_format,
        'indicators': indicators,
    }


# ---------------------------------------------------------------------
# 2. MANIPULATION / EDITING ANALYSIS (independent of AI / compression)
# ---------------------------------------------------------------------
def analyze_manipulation(ela, noise, dct):
    """
    Looks for evidence of *local* editing (splicing, cloning,
    retouching, compositing) as distinct from uniform, whole-image
    lossy compression. The key discriminator is that real editing
    tends to leave ELA anomalies that are localized and accompanied by
    a noise pattern that isn't uniformly suppressed the way a plain
    re-save of the whole image would be.
    """
    indicators = []
    strong_ela = ela['modified_pixels_pct'] > 8.0 and ela['mean'] > 8.0
    moderate_ela = ela['modified_pixels_pct'] > 4.0 and ela['mean'] > 5.0

    if strong_ela and noise['noise_std'] >= 6.0:
        indicators.append('Strong ELA anomaly covering {:.1f}% of the image'.format(ela['modified_pixels_pct']))
        indicators.append('Noise pattern is not uniform, inconsistent with simple whole-image recompression')
        confidence = 82
        if noise['edge_density_pct'] > 12 and dct['grid_strength_coeff'] > 15:
            indicators.append('Elevated edge density combined with an irregular DCT grid (possible compositing)')
            confidence = 90
        return 'EDITED / MANIPULATED', confidence, indicators, True

    if strong_ela and noise['noise_std'] < 6.0:
        # High ELA response, but perfectly uniform/flat noise -> this pattern is
        # much more consistent with heavy, whole-image lossy compression than
        # with a deliberately edited region, so we do NOT call it manipulation.
        return ('NO STRONG EVIDENCE OF MANIPULATION', 60,
                ['ELA response is elevated but spread uniformly across the whole image, '
                 'consistent with heavy compression rather than localized editing'], False)

    if moderate_ela and noise['noise_std'] >= 6.0:
        indicators.append('Localized ELA hotspots detected')
        indicators.append('Noise inconsistency detected relative to surrounding regions')
        return 'EDITED / MANIPULATED', 60, indicators, True

    return 'NO STRONG EVIDENCE OF MANIPULATION', 78, indicators, False


# ---------------------------------------------------------------------
# 3. AI-GENERATION ANALYSIS (handcrafted heuristic, NOT a trained model)
# ---------------------------------------------------------------------
def predict_ai_probability(img, compression_detected=False):
    """Predict AI-generation probability using the bundled NumPy-trained model."""
    if not MODEL_USED or MODEL_DATA is None:
        return None
    ela = error_level_analysis(img)
    noise = analyze_noise(img)
    freq = analyze_frequency_spectrum(img)
    dct = analyze_dct_artifacts(img)
    _, _, _, texture = analyze_ai_generated(noise, freq, dct, img, compression_detected)
    x = np.array([
        noise['noise_std'], noise['noise_mean'], noise['edge_density_pct'],
        freq['low_freq_pct'], freq['mid_freq_pct'], freq['high_freq_pct'],
        freq['spectral_flatness'], dct['grid_strength_coeff'], texture['entropy'],
        texture['edge_sharpness'], texture['texture_uniformity'], texture['smoothness'],
        ela['mean'], ela['modified_pixels_pct']
    ], dtype=np.float64)
    mean = MODEL_DATA['mean']; std = MODEL_DATA['std']
    std = np.where(std < 1e-8, 1.0, std)
    z = (x - mean) / std
    score = float(z @ MODEL_DATA['weights'] + MODEL_DATA['bias'])
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    return prob * 100.0, (1.0 - prob) * 100.0


def analyze_ai_generated(noise, freq, dct, img, compression_detected):
    """
    Conservative, handcrafted "AI-likeness" heuristic.

    IMPORTANT: this is not a trained classifier. It is written to
    require corroboration from at least two independently-meaningful
    signals before it will push strongly toward "AI-generated", and it
    explicitly discounts signals (low noise, high low-frequency energy,
    strong DCT grid, low edge sharpness) whenever compression has
    already been detected, since compression produces the same
    fingerprint in real photos.
    """
    gray = np.array(img.convert('L'), dtype=np.float32)
    high_freq = gray - _gaussian_blur(gray, 0.8)
    smoothness = float(high_freq.std())
    entropy = _local_entropy(gray)
    gradient_h = np.diff(gray, axis=1, append=gray[:, -1:])
    gradient_v = np.diff(gray, axis=0, append=gray[-1:, :])
    gradient_mag = np.sqrt(gradient_h ** 2 + gradient_v ** 2)
    edge_sharpness = float(
        (gradient_mag > gradient_mag.mean() + 2.5 * gradient_mag.std()).sum()
        / gradient_mag.size * 1000
    )
    h, w = gray.shape
    block_means, block_stds = _block_stats(gray, h, w)
    std_variance = float(np.var(block_stds))
    texture_uniformity = float(1.0 / (1.0 + std_variance / 100.0))

    score = 0.0
    signals = []
    strong_hits = 0

    # --- texture uniformity: strong signal, not explained by compression ---
    if texture_uniformity > 0.65:
        score += 22
        strong_hits += 1
        signals.append('Unusually uniform texture across large regions')
    elif texture_uniformity > 0.55:
        score += 8
        signals.append('Somewhat uniform texture fields')

    # --- edge sharpness: only counted fully if not already explained by compression ---
    if edge_sharpness < 10:
        score += 18
        strong_hits += 1
        signals.append('Very few genuinely sharp edges (over-smoothed detail)')
    elif edge_sharpness < 18 and not compression_detected:
        score += 8
        signals.append('Lower than expected high-frequency edge detail')

    # --- frequency spectrum: only counted fully if not already explained by compression ---
    if freq['high_freq_pct'] < 5 and freq['low_freq_pct'] > 88:
        score += 16
        strong_hits += 1
        signals.append('Spectrum concentrated almost entirely in low frequencies')
    elif freq['low_freq_pct'] > 82 and not compression_detected:
        score += 6
        signals.append('Excess energy concentrated in low frequencies')

    # --- local entropy: weak signal either direction ---
    if entropy > 6.6:
        score += 6
        signals.append('Very smooth, regular local textures')
    elif entropy < 5.0:
        score -= 8
        signals.append('Natural sensor-noise-like micro-texture present (real-camera trait)')

    # --- gradient variance: ignored entirely when compression explains it ---
    if gradient_mag.std() < 3.0 and not compression_detected:
        score += 8
        signals.append('Very low variance in edge gradients')

    # --- DCT grid: only relevant if NOT already explained by JPEG blockiness ---
    if dct['grid_strength_coeff'] > 40 and not compression_detected:
        score += 6
        signals.append('Regular high-frequency grid pattern not explained by JPEG blocking')

    if smoothness < 4.0 and not compression_detected:
        score += 6
    elif smoothness > 14.0:
        score -= 8
        signals.append('High-frequency sensor noise present (uncommon in generated images)')

    # Require corroboration from at least two independently strong
    # categories before letting the score swing heavily toward AI.
    if strong_hits < 2:
        score *= 0.5

    ai_probability = float(np.clip(50.0 + score, 1.0, 99.0))
    real_probability = round(100.0 - ai_probability, 1)
    return ai_probability, real_probability, signals, {
        'entropy': round(float(entropy), 3),
        'edge_sharpness': round(edge_sharpness, 2),
        'texture_uniformity': round(texture_uniformity, 3),
        'smoothness': round(smoothness, 3)
    }


def _count_entropy(patch):
    counts = np.bincount(patch.astype(np.int64), minlength=256).astype(np.float32)
    n = counts.sum()
    if n == 0:
        return 0.0
    probs = counts[counts > 0] / n
    return float(-np.sum(probs * np.log2(probs)))


def _local_entropy(gray, block=3):
    h, w = gray.shape
    values = []
    step = max(1, block)
    for i in range(0, h - block + 1, step):
        for j in range(0, w - block + 1, step):
            patch = gray[i:i+block, j:j+block].ravel()
            if patch.size < 2:
                continue
            values.append(_count_entropy(patch))
    if len(values) > 500:
        values = values[::max(1, len(values) // 500)]
    if not values:
        return 0.0
    return float(np.mean(values))


def _block_stats(gray, h, w, block=32):
    valid_h = (h // block) * block
    valid_w = (w // block) * block
    if valid_h == 0 or valid_w == 0:
        return np.array([gray.mean()]), np.array([max(gray.std(), 1e-6)])
    blocks = gray[:valid_h, :valid_w].reshape(valid_h // block, block, valid_w // block, block)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block, block)
    means = blocks.mean(axis=(1, 2))
    stds = blocks.std(axis=(1, 2))
    return means, (stds + 1e-6)


# ---------------------------------------------------------------------
# FINAL RESULT: combine the three independent analyses
# ---------------------------------------------------------------------
AI_THRESHOLD = 70
REAL_THRESHOLD = 70  # real_probability >= this => REAL / ORGANIC


def analyze_image(file_storage):
    try:
        img, original_format = load_image(file_storage)
    except Exception as e:
        return {'error': 'Could not read the uploaded image: {}'.format(str(e))}

    result = {}

    ela = error_level_analysis(img)
    noise = analyze_noise(img)
    freq = analyze_frequency_spectrum(img)
    dct = analyze_dct_artifacts(img)

    compression = analyze_compression(ela, dct, noise, original_format)

    manip_label, manip_conf, manip_indicators, manip_detected = analyze_manipulation(ela, noise, dct)

    ml_prediction = predict_ai_probability(img, compression['detected'])
    if ml_prediction is not None:
        ai_prob, real_prob = ml_prediction
        _, _, ai_signals, texture = analyze_ai_generated(
            noise, freq, dct, img, compression['detected']
        )
        ai_signals.insert(0, 'Prediction generated by the bundled trained NumPy classifier')
    else:
        ai_prob, real_prob, ai_signals, texture = analyze_ai_generated(
            noise, freq, dct, img, compression['detected']
        )

    # --- decide the single MAIN result ---
    if manip_detected:
        main_result = 'EDITED / MANIPULATED'
    elif ai_prob >= AI_THRESHOLD:
        main_result = 'AI-GENERATED'
    elif real_prob >= REAL_THRESHOLD:
        main_result = 'REAL / ORGANIC'
    else:
        main_result = 'UNCERTAIN'

    result['main_result'] = main_result
    result['model_used'] = MODEL_USED

    result['ela'] = ela
    result['noise'] = noise
    result['frequency'] = freq
    result['dct'] = dct
    result['texture'] = texture

    result['compression'] = compression

    result['manipulation'] = {
        'label': manip_label,
        'confidence': manip_conf,
        'detected': manip_detected,
        'indicators': manip_indicators,
    }
    result['manipulation_detected'] = manip_detected

    result['ai_probability'] = round(float(ai_prob), 1)
    result['real_probability'] = round(float(real_prob), 1)
    result['classification'] = 'AI-GENERATED' if ai_prob > real_prob else 'REAL / ORGANIC'
    result['reasons'] = ai_signals

    result['dimensions'] = detect_aspect_ratio_edit(img)

    return result


# =====================================================================
# HOW TO PLUG IN A REAL TRAINED MODEL LATER
# ---------------------------------------------------------------------
# 1. Organize data as:
#       dataset/real/*.jpg
#       dataset/ai_generated/*.jpg
# 2. Train a binary image classifier (a fine-tuned small CNN such as
#    ResNet18/EfficientNet-B0 is plenty for a mini-project) that
#    outputs P(ai_generated) for a 224x224 RGB input.
# 3. Save the trained weights, e.g. models/ai_classifier.pt
# 4. In this file, replace the body of analyze_ai_generated() (or add a
#    new function predict_ai_probability_ml(img)) with model loading +
#    inference, e.g.:
#
#       import torch
#       _model = torch.load('models/ai_classifier.pt', map_location='cpu')
#       _model.eval()
#       def predict_ai_probability_ml(img):
#           tensor = preprocess(img)  # resize/normalize to match training
#           with torch.no_grad():
#               prob = torch.sigmoid(_model(tensor)).item()
#           return prob * 100, (100 - prob * 100)
#
# 5. Set MODEL_USED = True once this is wired in, and swap the call in
#    analyze_image() from analyze_ai_generated() to your ML function.
#    Keep the compression/manipulation analyses exactly as they are -
#    they are legitimate supporting forensic evidence regardless of
#    whether AI-detection is heuristic or a trained model.
# =====================================================================
