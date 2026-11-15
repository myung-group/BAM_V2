"""Small helpers for the Korean methods + results PDF (reportlab + matplotlib mathtext equations).

Fonts: NanumGothic (text) and DejaVu Sans (symbols NanumGothic lacks: Angstrom, Greek, arrows, ...).
Equations: matplotlib mathtext (Computer Modern), rendered once to PNG and cached by a hash of the source.
"""
import hashlib
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

INK, MUTED, RULE, HEAD_BG, ACCENT = '#0b0b0b', '#52514e', '#c9c8c2', '#eef1f5', '#1f4e8c'
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
FRAME_W = PAGE_W - 2 * MARGIN


class Kit:
    def __init__(self, font_dir, asset_dir):
        self.assets = asset_dir
        os.makedirs(asset_dir, exist_ok=True)
        dejavu = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
        pdfmetrics.registerFont(TTFont('Nanum', os.path.join(font_dir, 'NanumGothic-Regular.ttf')))
        pdfmetrics.registerFont(TTFont('NanumB', os.path.join(font_dir, 'NanumGothic-Bold.ttf')))
        pdfmetrics.registerFont(TTFont('DejaVu', os.path.join(dejavu, 'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('DejaVuMono', os.path.join(dejavu, 'DejaVuSansMono.ttf')))
        self._have = set(pdfmetrics.getFont('Nanum').face.charToGlyph)
        plt.rcParams['mathtext.fontset'] = 'cm'
        S = ParagraphStyle
        self.st = {
            'title': S('title', fontName='NanumB', fontSize=21, leading=28, textColor=INK, spaceAfter=4),
            'sub': S('sub', fontName='Nanum', fontSize=10.5, leading=15, textColor=MUTED, spaceAfter=10),
            'h1': S('h1', fontName='NanumB', fontSize=15, leading=20, textColor=ACCENT, spaceBefore=14, spaceAfter=6, keepWithNext=1),
            'h2': S('h2', fontName='NanumB', fontSize=12, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4, keepWithNext=1),
            'h3': S('h3', fontName='NanumB', fontSize=10.5, leading=15, textColor=INK, spaceBefore=6, spaceAfter=2, keepWithNext=1),
            'p': S('p', fontName='Nanum', fontSize=9.8, leading=15.2, textColor=INK, spaceAfter=5),
            'b': S('b', fontName='Nanum', fontSize=9.8, leading=15.2, textColor=INK, leftIndent=12, bulletIndent=2,
                   spaceAfter=2.5),
            'cap': S('cap', fontName='Nanum', fontSize=8.6, leading=12.5, textColor=MUTED, spaceBefore=3, spaceAfter=8),
            'cell': S('cell', fontName='Nanum', fontSize=8.6, leading=12, textColor=INK),
            'cellb': S('cellb', fontName='NanumB', fontSize=8.6, leading=12, textColor=INK),
            'eqn': S('eqn', fontName='Nanum', fontSize=9, leading=12, textColor=MUTED, alignment=2),
            'note': S('note', fontName='Nanum', fontSize=8.8, leading=13.2, textColor=MUTED, spaceAfter=5),
        }
        self.story, self.n_eq, self.n_fig, self.n_tab = [], 0, 0, 0

    # ------------------------------------------------------------------ text
    def sym(self, text):
        """Wrap the characters NanumGothic has no glyph for in the DejaVu font (outside markup tags)."""
        out, run = [], []

        def flush():
            if run:
                out.append('<font name="DejaVu">' + ''.join(run) + '</font>'); run.clear()
        for part in re.split(r'(<[^>]+>)', text):
            if part.startswith('<') and part.endswith('>'):
                flush(); out.append(part); continue
            for ch in part:
                if ord(ch) > 126 and ord(ch) not in self._have:
                    run.append(ch)
                else:
                    flush(); out.append(ch)
            flush()
        return ''.join(out)

    def add(self, *flowables):
        self.story.extend(flowables)

    def title(self, text, sub=None):
        self.add(Paragraph(self.sym(text), self.st['title']))
        if sub:
            self.add(Paragraph(self.sym(sub), self.st['sub']))

    def h1(self, text):
        self.add(Paragraph(self.sym(text), self.st['h1']))

    def h2(self, text):
        self.add(Paragraph(self.sym(text), self.st['h2']))

    def h3(self, text):
        self.add(Paragraph(self.sym(text), self.st['h3']))

    def p(self, text, style='p'):
        self.add(Paragraph(self.sym(text), self.st[style]))

    def note(self, text):
        self.p(text, 'note')

    def bullets(self, items):
        for it in items:
            self.add(Paragraph(self.sym(it), self.st['b'], bulletText='•'))
        self.add(Spacer(1, 3))

    def gap(self, h=6):
        self.add(Spacer(1, h))

    def page(self):
        self.add(PageBreak())

    def need(self, points):
        """Start a new page unless `points` of height are left (keeps a heading with the block after it)."""
        self.add(CondPageBreak(points))

    # ------------------------------------------------------------------ equations
    def _eq_png(self, tex, fontsize):
        key = hashlib.sha1(f'{fontsize}|{tex}'.encode()).hexdigest()[:16]
        path = os.path.join(self.assets, f'eq_{key}.png')
        if not os.path.exists(path):
            fig = plt.figure(figsize=(0.1, 0.1))
            fig.text(0, 0, f'${tex}$', fontsize=fontsize, color=INK)
            fig.savefig(path, dpi=260, bbox_inches='tight', pad_inches=0.03, transparent=True)
            plt.close(fig)
        with PILImage.open(path) as im:
            w, h = im.size
        return path, w * 72.0 / 260, h * 72.0 / 260

    def eq(self, tex, fontsize=11.5, number=True, max_w=None):
        """Displayed equation, numbered at the right.  A list of strings = several lines of one equation."""
        lines = tex if isinstance(tex, (list, tuple)) else [tex]
        max_w = max_w or (FRAME_W - 16 * mm)
        imgs = []
        for t in lines:
            path, w, h = self._eq_png(t, fontsize)
            if w > max_w:
                h, w = h * max_w / w, max_w
            imgs.append(Image(path, width=w, height=h))
        label = ''
        if number:
            self.n_eq += 1
            label = f'({self.n_eq})'
        rows = [[im, Paragraph(label if i == len(imgs) - 1 else '', self.st['eqn'])] for i, im in enumerate(imgs)]
        t = Table(rows, colWidths=[FRAME_W - 14 * mm, 14 * mm])
        t.setStyle(TableStyle([('ALIGN', (0, 0), (0, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                               ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]))
        self.add(KeepTogether([t, Spacer(1, 3)]))
        return self.n_eq

    def inline(self, tex, fontsize=9.8):
        """Inline math as an <img> tag for use inside a paragraph string."""
        path, w, h = self._eq_png(tex, fontsize)
        return f'<img src="{path}" width="{w:.1f}" height="{h:.1f}" valign="-{max(0.0, h * 0.28):.1f}"/>'

    # ------------------------------------------------------------------ tables and figures
    def table(self, rows, widths=None, caption=None, header=True, align_right_from=1, bold_first_col=False,
              font=None):
        """rows: list of lists of strings (markup allowed).  widths: fractions of the frame width."""
        n = len(rows[0])
        widths = [FRAME_W * w for w in (widths or [1.0 / n] * n)]
        cell, cellb = self.st['cell'], self.st['cellb']
        if font:
            cell = ParagraphStyle('c2', parent=cell, fontSize=font, leading=font * 1.4)
            cellb = ParagraphStyle('c2b', parent=cellb, fontSize=font, leading=font * 1.4)
        data = []
        for i, r in enumerate(rows):
            row = []
            for j, c in enumerate(r):
                st = cellb if (header and i == 0) or (bold_first_col and j == 0) else cell
                if j >= align_right_from and align_right_from >= 0:
                    st = ParagraphStyle('r', parent=st, alignment=2)
                row.append(Paragraph(self.sym(str(c)), st))
            data.append(row)
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 3),
                 ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('LEFTPADDING', (0, 0), (-1, -1), 5),
                 ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                 ('LINEBELOW', (0, 0), (-1, -1), 0.4, colors.HexColor(RULE)),
                 ('LINEABOVE', (0, 0), (-1, 0), 0.8, colors.HexColor(INK)),
                 ('LINEBELOW', (0, -1), (-1, -1), 0.8, colors.HexColor(INK))]
        if header:
            style += [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
                      ('LINEBELOW', (0, 0), (-1, 0), 0.8, colors.HexColor(INK))]
        t.setStyle(TableStyle(style))
        block = []
        if caption:
            self.n_tab += 1
            block.append(Paragraph(self.sym(f'<font name="NanumB">표 {self.n_tab}.</font> {caption}'), self.st['cap']))
        block.append(t)
        block.append(Spacer(1, 8))
        self.add(KeepTogether(block) if len(rows) <= 22 else block[0], *([] if len(rows) <= 22 else block[1:]))

    def figure(self, path, caption=None, width=1.0):
        with PILImage.open(path) as im:
            w, h = im.size
        fw = FRAME_W * width
        fh = fw * h / w
        max_h = PAGE_H - 2 * MARGIN - 60
        if fh > max_h:
            fw, fh = fw * max_h / fh, max_h
        block = [Image(path, width=fw, height=fh)]
        if caption:
            self.n_fig += 1
            block.append(Paragraph(self.sym(f'<font name="NanumB">그림 {self.n_fig}.</font> {caption}'), self.st['cap']))
        self.add(KeepTogether(block))

    # ------------------------------------------------------------------ build
    def build(self, out, title, author='', footer=''):
        def decorate(canvas, doc):
            canvas.saveState()
            canvas.setFont('Nanum', 8)
            canvas.setFillColor(colors.HexColor(MUTED))
            canvas.drawString(MARGIN, 10 * mm, footer)
            canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, str(doc.page))
            canvas.restoreState()
        doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                                bottomMargin=MARGIN, title=title, author=author)
        doc.build(self.story, onFirstPage=decorate, onLaterPages=decorate)
        return out
