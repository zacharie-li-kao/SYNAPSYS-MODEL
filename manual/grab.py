"""Shared helper for the manual's figures: grab a Tk window's pixels."""
import ctypes, os
from ctypes import wintypes
from PIL import ImageGrab

if os.environ.get("SYN_SCALE", "1.5") != "auto":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

import customtkinter as ctk

# One DPI authority: Windows (per-monitor, set above). Without this ctk applies
# its own factor on top and Tk's geometry no longer matches the pixels on screen.
# SYN_SCALE=auto leaves customtkinter's own handling alone (matplotlib canvases then
# size correctly, but Tk coordinates no longer equal screen pixels: no markers).
if os.environ.get("SYN_SCALE", "1.5") != "auto":
    ctk.deactivate_automatic_dpi_awareness()
    SCALE = float(os.environ.get("SYN_SCALE", "1.5"))
    ctk.set_widget_scaling(SCALE)
    ctk.set_window_scaling(SCALE)

_u = ctypes.windll.user32


def client_bbox(root):
    """Client area of the toplevel in physical screen pixels (Win32, not Tk)."""
    hwnd = _u.GetParent(root.winfo_id()) or root.winfo_id()
    rc = wintypes.RECT()
    _u.GetClientRect(hwnd, ctypes.byref(rc))
    pt = wintypes.POINT(0, 0)
    _u.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, pt.x + rc.right, pt.y + rc.bottom


def _print_window(hwnd):
    """The window's own pixels (PrintWindow) and its screen rectangle."""
    from PIL import Image
    rc = wintypes.RECT(); _u.GetWindowRect(hwnd, ctypes.byref(rc))
    w, h = rc.right - rc.left, rc.bottom - rc.top
    gdi = ctypes.windll.gdi32
    hdc = _u.GetWindowDC(hwnd); mdc = gdi.CreateCompatibleDC(hdc)
    bmp = gdi.CreateCompatibleBitmap(hdc, w, h); gdi.SelectObject(mdc, bmp)
    if not _u.PrintWindow(hwnd, mdc, 2):  # PW_RENDERFULLCONTENT
        raise RuntimeError("PrintWindow failed")

    class BIH(ctypes.Structure):
        _fields_ = [("s", wintypes.DWORD), ("w", wintypes.LONG), ("h", wintypes.LONG),
                    ("p", wintypes.WORD), ("b", wintypes.WORD), ("c", wintypes.DWORD),
                    ("si", wintypes.DWORD), ("x", wintypes.LONG), ("y", wintypes.LONG),
                    ("u", wintypes.DWORD), ("i", wintypes.DWORD)]
    bi = BIH(ctypes.sizeof(BIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
    gdi.DeleteObject(bmp); gdi.DeleteDC(mdc); _u.ReleaseDC(hwnd, hdc)
    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")
    return img, rc


def grab(root, path=None, crop=None):
    """The toplevel's client area as a PIL image (saved to `path` if given);
    crop=(x0,y0,x1,y1) in client pixels."""
    for _ in range(5):
        root.update(); root.update_idletasks()
    hwnd = _u.GetParent(root.winfo_id()) or root.winfo_id()
    img, rc = _print_window(hwnd)
    cx0, cy0, cx1, cy1 = client_bbox(root)
    img = img.crop((cx0 - rc.left, cy0 - rc.top, cx1 - rc.left, cy1 - rc.top))
    if crop:
        img = img.crop(crop)
    if path:
        img.save(path)
    return img


def trim_black(img):
    """PrintWindow leaves the invisible resize frame of a dialog black."""
    img = img.convert("RGB")
    row = int(img.height * 0.6)
    xs = [x for x in range(img.width) if img.getpixel((x, row)) != (0, 0, 0)]
    img = img.crop((min(xs), 0, max(xs) + 1, img.height))
    return img.crop(img.getbbox())


def grab_dialog(title, path, answer=None):
    """Save a native message box (found by its title) as PNG, then answer it
    (answer = the button id, e.g. 6 = Yes) or close it."""
    hwnd = _u.FindWindowW("#32770", title)
    if not hwnd:
        raise RuntimeError(f"no dialog titled {title!r}")
    img, _ = _print_window(hwnd)
    img = trim_black(img)
    img.save(path)
    if answer is None:
        _u.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
    else:
        _u.PostMessageW(hwnd, 0x0111, answer, 0)  # WM_COMMAND
    return img.size
