"""
thai_text.py  –  วาดข้อความภาษาไทยบน OpenCV frame ด้วย Pillow
เพราะ cv2.putText รองรับเฉพาะ ASCII เท่านั้น
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from typing import Tuple

# ── โหลด Pillow ──────────────────────────────────────────────────────────────
try:
    from PIL import ImageFont, ImageDraw, Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ── ค้นหา font ที่รองรับภาษาไทย ──────────────────────────────────────────────
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\cordia.ttc",
    r"C:\Windows\Fonts\angsana.ttc",
    "/usr/share/fonts/truetype/thai/Garuda.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
]

_font_cache: dict = {}


def _get_font(size: int) -> "ImageFont.FreeTypeFont | None":
    if not _PIL_OK:
        return None
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                fnt = ImageFont.truetype(path, size)
                _font_cache[size] = fnt
                return fnt
            except Exception:
                continue
    _font_cache[size] = None
    return None


Color = Tuple[int, int, int]  # BGR


def put_thai(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int = 20,
    color: Color = (240, 240, 240),
    thickness: int = 2,
) -> None:
    """
    วาดข้อความ (รองรับภาษาไทย) ลงบน OpenCV frame (in-place)

    Parameters
    ----------
    frame     : BGR numpy array
    text      : ข้อความที่จะวาด
    pos       : (x, y) ตำแหน่งซ้ายล่างของบรรทัด  (เหมือน cv2.putText)
    font_size : ขนาด font (pixels)
    color     : BGR tuple
    thickness : ไม่ใช้งานโดยตรงกับ PIL แต่เก็บไว้เพื่อ API compatibility
    """
    font = _get_font(font_size)
    if font is None or not _PIL_OK:
        # Fallback: ASCII only via OpenCV
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 32,
                    color, thickness, cv2.LINE_AA)
        return

    # แปลง BGR → RGB เพื่อใช้กับ PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    # PIL สี = RGB
    pil_color = (color[2], color[1], color[0])

    # PIL วาดจากมุมซ้ายบน  ─  แก้ offset จาก pos (ซ้ายล่าง)
    x, y_bottom = pos
    bbox = font.getbbox("Ag")      # ประมาณความสูงบรรทัด
    line_height = bbox[3] - bbox[1]
    y_top = y_bottom - line_height - 2

    draw.text((x, y_top), text, font=font, fill=pil_color)

    # แปลงกลับ RGB → BGR แล้วเขียนทับ frame เดิม
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)


def draw_hud_box(
    frame: np.ndarray,
    lines: list,          # list of (text, bgr_color)
    top: int = 4,
    box_w: int = 320,
    font_size: int = 20,
    line_h: int = 32,
    pad_x: int = 12,
    pad_y: int = 10,
    border_color: Color = (0, 180, 180),
) -> None:
    """
    วาดกล่อง HUD โปร่งแสง + เส้นขอบ + ข้อความภาษาไทย
    """
    box_h = pad_y * 2 + line_h * len(lines)

    # วาดพื้นหลังโปร่งแสง
    overlay = frame.copy()
    cv2.rectangle(overlay, (4, top), (4 + box_w, top + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # เส้นขอบ
    cv2.rectangle(frame, (4, top), (4 + box_w, top + box_h), border_color, 1)

    # วาดข้อความแต่ละบรรทัด
    y = top + pad_y + line_h - 4
    for text, color in lines:
        put_thai(frame, text, (4 + pad_x, y), font_size=font_size, color=color)
        y += line_h
