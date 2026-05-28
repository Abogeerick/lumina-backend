"""
Curated foundation shade database covering the full Fitzpatrick spectrum (I-VI).
Hex values are approximations from publicly available brand swatches.
Lab values are computed at import time for efficient matching.
"""
import math


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_lab(r, g, b):
    # sRGB -> linear RGB
    def _linearise(c):
        c /= 255.0
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    rl, gl, bl = _linearise(r), _linearise(g), _linearise(b)

    # linear RGB -> XYZ (D65)
    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    # Normalise by D65 reference white
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    # XYZ -> Lab
    def _f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16 / 116)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return round(L, 2), round(a, 2), round(b_, 2)


_RAW_SHADES = [
    # ===== Fenty Beauty Pro Filt'r Soft Matte =====
    {'brand': 'Fenty Beauty', 'product': "Pro Filt'r Soft Matte", 'shade': '100',
     'hex': '#F2DCC4', 'undertone': 'cool', 'fitzpatrick': 1,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'Fenty Beauty', 'product': "Pro Filt'r Soft Matte", 'shade': '150',
     'hex': '#E8C8A6', 'undertone': 'neutral', 'fitzpatrick': 2,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'Fenty Beauty', 'product': "Pro Filt'r Soft Matte", 'shade': '220',
     'hex': '#D8AE85', 'undertone': 'neutral', 'fitzpatrick': 3,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'Fenty Beauty', 'product': "Pro Filt'r Soft Matte", 'shade': '290',
     'hex': '#C29773', 'undertone': 'warm', 'fitzpatrick': 3,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'Fenty Beauty', 'product': "Pro Filt'r Soft Matte", 'shade': '360',
     'hex': '#A87B5A', 'undertone': 'neutral', 'fitzpatrick': 4,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'Fenty Beauty', 'product': "Pro Filt'r Soft Matte", 'shade': '410',
     'hex': '#8B5E40', 'undertone': 'warm', 'fitzpatrick': 5,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'Fenty Beauty', 'product': "Pro Filt'r Soft Matte", 'shade': '470',
     'hex': '#6B4530', 'undertone': 'cool', 'fitzpatrick': 5,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'Fenty Beauty', 'product': "Pro Filt'r Soft Matte", 'shade': '498',
     'hex': '#3F2418', 'undertone': 'neutral', 'fitzpatrick': 6,
     'comedogenic': False, 'pigmentation_friendly': True},

    # ===== Maybelline Fit Me Matte+Poreless =====
    {'brand': 'Maybelline', 'product': 'Fit Me Matte+Poreless', 'shade': '110 Porcelain',
     'hex': '#F5DBC0', 'undertone': 'cool', 'fitzpatrick': 1,
     'comedogenic': False, 'pigmentation_friendly': False},
    {'brand': 'Maybelline', 'product': 'Fit Me Matte+Poreless', 'shade': '125 Nude Beige',
     'hex': '#ECC8A5', 'undertone': 'warm', 'fitzpatrick': 2,
     'comedogenic': False, 'pigmentation_friendly': False},
    {'brand': 'Maybelline', 'product': 'Fit Me Matte+Poreless', 'shade': '220 Natural Beige',
     'hex': '#D9B088', 'undertone': 'neutral', 'fitzpatrick': 3,
     'comedogenic': False, 'pigmentation_friendly': False},
    {'brand': 'Maybelline', 'product': 'Fit Me Matte+Poreless', 'shade': '310 Sun Beige',
     'hex': '#BD8E64', 'undertone': 'warm', 'fitzpatrick': 4,
     'comedogenic': False, 'pigmentation_friendly': False},
    {'brand': 'Maybelline', 'product': 'Fit Me Matte+Poreless', 'shade': '330 Toffee',
     'hex': '#A2754D', 'undertone': 'warm', 'fitzpatrick': 4,
     'comedogenic': False, 'pigmentation_friendly': False},
    {'brand': 'Maybelline', 'product': 'Fit Me Matte+Poreless', 'shade': '360 Mocha',
     'hex': '#7D5436', 'undertone': 'neutral', 'fitzpatrick': 5,
     'comedogenic': False, 'pigmentation_friendly': False},
    {'brand': 'Maybelline', 'product': 'Fit Me Matte+Poreless', 'shade': '375 Java',
     'hex': '#5B3A23', 'undertone': 'warm', 'fitzpatrick': 6,
     'comedogenic': False, 'pigmentation_friendly': False},
    {'brand': 'Maybelline', 'product': 'Fit Me Matte+Poreless', 'shade': '380 Rich Espresso',
     'hex': '#3D2515', 'undertone': 'cool', 'fitzpatrick': 6,
     'comedogenic': False, 'pigmentation_friendly': False},

    # ===== MAC Studio Fix Fluid =====
    {'brand': 'MAC', 'product': 'Studio Fix Fluid', 'shade': 'NW10',
     'hex': '#F0D6BC', 'undertone': 'cool', 'fitzpatrick': 1,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'MAC', 'product': 'Studio Fix Fluid', 'shade': 'NC15',
     'hex': '#EFC9A0', 'undertone': 'warm', 'fitzpatrick': 2,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'MAC', 'product': 'Studio Fix Fluid', 'shade': 'NW25',
     'hex': '#D2A57E', 'undertone': 'cool', 'fitzpatrick': 3,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'MAC', 'product': 'Studio Fix Fluid', 'shade': 'NC35',
     'hex': '#BE8E63', 'undertone': 'warm', 'fitzpatrick': 4,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'MAC', 'product': 'Studio Fix Fluid', 'shade': 'NC42',
     'hex': '#A37448', 'undertone': 'warm', 'fitzpatrick': 4,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'MAC', 'product': 'Studio Fix Fluid', 'shade': 'NW45',
     'hex': '#7F5234', 'undertone': 'cool', 'fitzpatrick': 5,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'MAC', 'product': 'Studio Fix Fluid', 'shade': 'NC55',
     'hex': '#5C3920', 'undertone': 'warm', 'fitzpatrick': 6,
     'comedogenic': False, 'pigmentation_friendly': True},
    {'brand': 'MAC', 'product': 'Studio Fix Fluid', 'shade': 'NW58',
     'hex': '#3A2113', 'undertone': 'cool', 'fitzpatrick': 6,
     'comedogenic': False, 'pigmentation_friendly': True},
]


def _build_shades():
    shades = []
    for s in _RAW_SHADES:
        r, g, b = _hex_to_rgb(s['hex'])
        L, a, b_ = _rgb_to_lab(r, g, b)
        shades.append({
            **s,
            'rgb': [r, g, b],
            'lab': [L, a, b_],
        })
    return shades


SHADES = _build_shades()
