"""Build manual/SYNAPSYS_manual.pdf from manual/chapters/*.md.

Markdown subset: '# ' headings, paragraphs, '- ' click sequences, ![caption](figures/x.png).
{x} in the text is the number of figures/x.png. The version on the cover is read from
Install_script.iss, so the manual cannot drift from the suite.

Figures (run from the project root; needs lupa, a display, and a seeded test suite made with
generate_test_data.py --seed 20260731 <dir> and saved as JSON by the data converter):
    python -m manual.fig_hub
    python -m manual.fig_char                               (simulated 2636B, about 3 minutes)
    SYN_SCALE=1.0 python -m manual.fig_conv <dir>
    SYN_SCALE=auto python -m manual.fig_model <suite.json>
"""
import os, re, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Image, KeepTogether, ListFlowable, ListItem,
                                PageBreak)
from PIL import Image as PILImage

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = r"C:\Windows\Fonts"
for name, f in (("Body", "segoeui.ttf"), ("Body-B", "segoeuib.ttf"),
                ("Body-I", "segoeuii.ttf"), ("Mono", "consola.ttf")):
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, f)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-B")

BODY = ParagraphStyle("b", fontName="Body", fontSize=10, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7)
H1 = ParagraphStyle("h", fontName="Body-B", fontSize=16, leading=20, spaceBefore=6, spaceAfter=10)
CAP = ParagraphStyle("c", fontName="Body-I", fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"),
                     spaceBefore=3, spaceAfter=10)
TITLE = ParagraphStyle("t", fontName="Body-B", fontSize=26, leading=32, spaceAfter=8)
SUB = ParagraphStyle("s", fontName="Body", fontSize=12, leading=16, textColor=colors.HexColor("#444444"))


def suite_version():
    """The suite version is the installer's; the manual must always match it."""
    root = os.path.dirname(HERE)
    iss = open(os.path.join(root, "Install_script.iss"), encoding="utf-8").read()
    suite = re.search(r'#define MyAppVersion "([^"]+)"', iss).group(1)
    ka = open(os.path.join(root, "keithley_analyser.py"), encoding="utf-8").read()
    char = re.search(r'__version__ = "([^"]+)"', ka).group(1)
    return suite, char


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"`([^`]+)`", r'<font name="Mono" size="9">\1</font>', t)


def load_chapters():
    """Concatenate chapters/*.md and number the figures in order of appearance;
    {name} in the text becomes the number of figures/name.png."""
    d = os.path.join(HERE, "chapters")
    text = "\n\n".join(open(os.path.join(d, f), encoding="utf-8").read().strip()
                       for f in sorted(os.listdir(d)) if f.endswith(".md"))
    names = re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\.png\)", text)
    nums = {n: i + 1 for i, n in enumerate(names)}
    missing = set(re.findall(r"\{([a-z0-9_]+)\}", text)) - set(nums)
    if missing:
        raise SystemExit(f"figure references without a figure: {sorted(missing)}")
    return re.sub(r"\{([a-z0-9_]+)\}", lambda m: str(nums[m.group(1)]), text)


def build(text, pdf_path):
    story, fig_n, bullets = [], 0, []
    width = A4[0] - 40 * mm

    def flush():
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(inline(b), BODY), leftIndent=14) for b in bullets],
                                      bulletType="1", bulletFontName="Body", leftIndent=14))
            bullets.clear()

    suite, char = suite_version()
    story += [Spacer(1, 60 * mm), Paragraph("SYNAPSYS", TITLE),
              Paragraph(f"User manual, version {suite}", SUB), Spacer(1, 4 * mm),
              Paragraph(f"Characterisation tool v{char}", SUB), PageBreak()]
    first = True
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        m = re.fullmatch(r"!\[(.*)\]\((.*)\)", block, re.S)
        if block.startswith("# "):
            flush()
            if not first:
                story.append(PageBreak())
            first = False
            story.append(Paragraph(inline(block[2:]), H1))
        elif m:
            flush()
            fig_n += 1
            p = os.path.join(HERE, m.group(2))
            w, h = PILImage.open(p).size
            # full-window shots span the text width; dialogs and panels keep a readable
            # size (0.6 pt per pixel at most) and are limited in height
            max_h = (A4[1] - 40 * mm) * 0.40
            scale = min(width / w, 0.5, max_h / h)
            iw = w * scale
            story.append(KeepTogether([Image(p, width=iw, height=iw * h / w, hAlign="CENTER"),
                                       Paragraph(f"Figure {fig_n}. " + inline(m.group(1)), CAP)]))
        elif block.startswith("- "):
            bullets += [l[2:] for l in block.split("\n")]
        else:
            flush()
            story.append(Paragraph(inline(" ".join(block.split("\n"))), BODY))
    flush()

    def footer(c, d):
        c.setFont("Body", 8); c.setFillColor(colors.grey)
        c.drawCentredString(A4[0] / 2, 12 * mm, str(d.page))

    doc = BaseDocTemplate(pdf_path, pagesize=A4, title=f"SYNAPSYS user manual v{suite}", author="Zacharie Jehl Li-Kao")
    doc.addPageTemplates([PageTemplate(id="p", onPage=footer, frames=[
        Frame(20 * mm, 20 * mm, width, A4[1] - 40 * mm, id="f")])])
    doc.build(story)


if __name__ == "__main__":
    build(load_chapters(), os.path.join(HERE, "SYNAPSYS_manual.pdf"))
