"""
Skin tone analysis pipeline:
  1. Detect skin pixels in YCrCb space
  2. Compute robust median Lab of skin pixels
  3. Derive ITA (Individual Typology Angle) -> Fitzpatrick type
  4. Match to foundation shades by Delta E (CIE76)
"""
import io
import math
import numpy as np
from PIL import Image

from foundation_shades import SHADES


# YCrCb skin-pixel thresholds (loose, widely cited)
SKIN_Y_MIN = 60
SKIN_CR_MIN, SKIN_CR_MAX = 133, 173
SKIN_CB_MIN, SKIN_CB_MAX = 77, 127


def _rgb_to_ycrcb(rgb):
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    return y, cr, cb


def _extract_skin_pixels(image_bytes):
    """Return median RGB of detected skin pixels, falling back to median of whole image."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Downsample for speed
    img.thumbnail((300, 300))
    arr = np.array(img)

    y, cr, cb = _rgb_to_ycrcb(arr)
    skin_mask = (
        (y > SKIN_Y_MIN) &
        (cr >= SKIN_CR_MIN) & (cr <= SKIN_CR_MAX) &
        (cb >= SKIN_CB_MIN) & (cb <= SKIN_CB_MAX)
    )

    skin_pixels = arr[skin_mask]
    detected_count = int(skin_pixels.shape[0])
    total_pixels = int(arr.shape[0] * arr.shape[1])
    skin_ratio = detected_count / total_pixels if total_pixels else 0

    if detected_count < 200:
        # Too few skin pixels — fall back to whole image median
        median = np.median(arr.reshape(-1, 3), axis=0)
        return tuple(int(v) for v in median), skin_ratio, False

    median = np.median(skin_pixels, axis=0)
    return tuple(int(v) for v in median), skin_ratio, True


def _rgb_to_lab(r, g, b):
    def _linearise(c):
        c /= 255.0
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    rl, gl, bl = _linearise(r), _linearise(g), _linearise(b)

    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    x /= 0.95047
    z /= 1.08883

    def _f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return L, a, b_


def _compute_ita(L, b):
    """ITA in degrees. b must be non-zero (it always is for human skin)."""
    if b == 0:
        return 0.0
    return math.degrees(math.atan((L - 50) / b))


def _fitzpatrick_from_ita(ita):
    """Standard Chardon classification mapped to Fitzpatrick I-VI."""
    if ita > 55:
        return 1, 'Very Light'
    if ita > 41:
        return 2, 'Light'
    if ita > 28:
        return 3, 'Intermediate'
    if ita > 10:
        return 4, 'Tan'
    if ita > -30:
        return 5, 'Brown'
    return 6, 'Deep'


def _detect_undertone(a, b):
    """Coarse undertone hint from Lab a*/b* values."""
    if b > 18 and a < 16:
        return 'warm'
    if b < 12:
        return 'cool'
    if a >= 16:
        return 'warm'
    return 'neutral'


def _delta_e_76(lab1, lab2):
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(lab1, lab2)))


def _match_shades(skin_lab, undertone, condition=None, top_n=6):
    candidates = []
    for shade in SHADES:
        distance = _delta_e_76(skin_lab, shade['lab'])

        # Boost shades that match undertone (subtract from distance)
        undertone_bonus = 0
        if shade['undertone'] == undertone:
            undertone_bonus = 4
        elif shade['undertone'] == 'neutral' or undertone == 'neutral':
            undertone_bonus = 1.5

        # Boost condition-appropriate shades
        condition_bonus = 0
        if condition == 'acne' and not shade.get('comedogenic', True):
            condition_bonus = 2.5
        elif condition == 'hyperpigmentation' and shade.get('pigmentation_friendly'):
            condition_bonus = 2.5

        score = distance - undertone_bonus - condition_bonus
        candidates.append({
            'shade': shade,
            'delta_e': round(distance, 2),
            'score': round(score, 2),
        })

    candidates.sort(key=lambda c: c['score'])
    return candidates[:top_n]


def analyse_skin_tone(image_bytes, condition=None):
    rgb, skin_ratio, detected = _extract_skin_pixels(image_bytes)
    r, g, b = rgb
    L, a, b_lab = _rgb_to_lab(r, g, b)
    ita = _compute_ita(L, b_lab)
    fitz_type, fitz_label = _fitzpatrick_from_ita(ita)
    undertone = _detect_undertone(a, b_lab)

    hex_color = '#{:02X}{:02X}{:02X}'.format(*rgb)
    matches = _match_shades([L, a, b_lab], undertone, condition=condition)

    return {
        'skin_tone': {
            'rgb': list(rgb),
            'hex': hex_color,
            'lab': {'L': round(L, 2), 'a': round(a, 2), 'b': round(b_lab, 2)},
            'ita': round(ita, 2),
            'fitzpatrick': {'type': fitz_type, 'label': fitz_label},
            'undertone': undertone,
            'skin_detection_confidence': round(min(skin_ratio * 100, 100), 1),
            'detected': detected,
        },
        'matches': [
            {
                'brand': m['shade']['brand'],
                'product': m['shade']['product'],
                'shade': m['shade']['shade'],
                'hex': m['shade']['hex'],
                'undertone': m['shade']['undertone'],
                'fitzpatrick': m['shade']['fitzpatrick'],
                'delta_e': m['delta_e'],
            }
            for m in matches
        ],
        'condition_filtered_for': condition,
    }
