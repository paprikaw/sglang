"""Editable PowerPoint text/shapes plus deterministic image/PDF previews."""
from pathlib import Path
import json
import textwrap

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

W, H, PPI = 1920, 1080, 144
BG, INK, MUTED = '#F4F7FC', '#18243D', '#61718A'
BLUE, TEAL, ORANGE = '#326FE9', '#009C88', '#D78131'
FONT_ROOT = Path('/tmp/pp_ep_ppt_assets')


def color(value):
    return RGBColor.from_string(value.lstrip('#'))


class Deck:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'previews').mkdir(exist_ok=True)
        self.prs = Presentation()
        self.prs.slide_width = Inches(40 / 3)
        self.prs.slide_height = Inches(7.5)
        self.pages, self.notes, self.bounds = [], [], []

    def slide(self, title, subtitle='', sources=(), notes=''):
        self.current = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.canvas = Image.new('RGB', (W, H), BG)
        self.draw = ImageDraw.Draw(self.canvas)
        self.current.background.fill.solid()
        self.current.background.fill.fore_color.rgb = color(BG)
        self.rect(0.55, 0.43, 0.08, 0.43, BLUE, radius=False)
        self.text(title, 0.78, 0.42, 11.8, 0.58, 29, bold=True)
        if subtitle:
            self.text(subtitle, 0.78, 1.08, 11.8, 0.5, 12, MUTED)
        self.text('PP / EP  |  固定 GPU 预算下的部署决策', 0.6, 7.14, 9.6, 0.2, 8.5, MUTED)
        self.text(f'{len(self.pages) + 1:02d}', 12.1, 7.10, 0.6, 0.28, 10, MUTED, align='right')
        source_text = '\n'.join(str(p) for p in sources)
        self.current.notes_slide.notes_text_frame.text = notes + '\n\n数据来源（绝对路径）：\n' + source_text
        self.notes.append(dict(slide=len(self.pages) + 1, title=title, notes=notes, sources=list(map(str, sources))))

    def rect(self, x, y, width, height, fill='#FFFFFF', radius=True, line=None):
        self._bounds(x, y, width, height, 'shape')
        box = (round(x * PPI), round(y * PPI), round((x + width) * PPI), round((y + height) * PPI))
        if radius:
            self.draw.rounded_rectangle(box, radius=16, fill=fill, outline=line)
        else:
            self.draw.rectangle(box, fill=fill, outline=line)
        shape = self.current.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
            Inches(x), Inches(y), Inches(width), Inches(height))
        if radius:
            shape.adjustments[0] = 0.08
        shape.fill.solid()
        shape.fill.fore_color.rgb = color(fill)
        if line:
            shape.line.color.rgb = color(line)
        else:
            shape.line.fill.background()

    def text(self, value, x, y, width, height, size=18, ink=INK, bold=False, align='left', wrap=True):
        self._bounds(x, y, width, height, 'text')
        font = ImageFont.truetype(str(FONT_ROOT / f'NotoSansCJKsc-{"Bold" if bold else "Regular"}.otf'), round(size * 2))
        lines = []
        max_width = width * PPI
        for paragraph in str(value).split('\n'):
            line = ''
            for char in paragraph:
                if wrap and line and self.draw.textlength(line + char, font=font) > max_width:
                    lines.append(line.rstrip())
                    line = char.lstrip()
                else:
                    line += char
            lines.append(line)
        line_height = size / 72 * PPI * 1.25
        if len(lines) * line_height > height * PPI + 3:
            raise ValueError(f'Text overflows: {value!r}: {len(lines)} lines in {height} inches')
        textbox = self.current.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
        frame = textbox.text_frame
        frame.clear()
        frame.word_wrap = False
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.vertical_anchor = MSO_ANCHOR.TOP
        for index, line in enumerate(lines):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = line
            paragraph.alignment = {'left': PP_ALIGN.LEFT, 'center': PP_ALIGN.CENTER, 'right': PP_ALIGN.RIGHT}[align]
            paragraph.space_before = paragraph.space_after = Pt(0)
            paragraph.line_spacing = Pt(size * 1.25)
            for run in paragraph.runs:
                run.font.name = 'Noto Sans CJK SC'
                run.font.size = Pt(size)
                run.font.bold = bold
                run.font.color.rgb = color(ink)
                properties = run._r.get_or_add_rPr()
                properties.set('lang', 'zh-CN')
                for font_kind in ['ea', 'cs']:
                    element = OxmlElement('a:' + font_kind)
                    element.set('typeface', 'Noto Sans CJK SC')
                    properties.append(element)
            length = self.draw.textlength(line, font=font)
            offset = (max_width - length) / 2 if align == 'center' else max_width - length if align == 'right' else 0
            self.draw.text((x * PPI + offset, y * PPI + index * line_height), line,
                           font=font, fill=ink, anchor='lt')

    def image(self, path, x, y, width, height):
        self._bounds(x, y, width, height, 'image')
        image = Image.open(path).convert('RGB')
        scale = min(width * PPI / image.width, height * PPI / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        px = round(x * PPI + (width * PPI - resized.width) / 2)
        py = round(y * PPI + (height * PPI - resized.height) / 2)
        self.canvas.paste(resized, (px, py))
        self.current.shapes.add_picture(str(path), Inches(px / PPI), Inches(py / PPI),
            width=Inches(resized.width / PPI), height=Inches(resized.height / PPI))

    def _bounds(self, x, y, width, height, kind):
        assert width > 0 and height > 0 and x >= 0 and y >= 0
        assert x + width <= 40 / 3 + .001 and y + height <= 7.5 + .001, (kind, x, y, width, height)
        self.bounds.append(dict(slide=len(self.pages) + 1, kind=kind, x=x, y=y, width=width, height=height))

    def finish_slide(self):
        number = len(self.pages) + 1
        self.canvas.save(self.directory / 'previews' / f'slide_{number:02d}.png')
        self.pages.append(self.canvas)

    def save(self, stem):
        self.prs.save(self.directory / f'{stem}.pptx')
        self.pages[0].save(self.directory / f'{stem}.pdf', save_all=True,
            append_images=self.pages[1:], resolution=PPI, quality=95)
        (self.directory / 'slide_notes.json').write_text(json.dumps(self.notes, ensure_ascii=False, indent=2) + '\n')
        (self.directory / 'layout_bounds.json').write_text(json.dumps(self.bounds, indent=2) + '\n')
        thumb_w, thumb_h = 640, 360
        rows = (len(self.pages) + 2) // 3
        sheet = Image.new('RGB', (thumb_w * 3, thumb_h * rows), '#DDE5F0')
        for index, page in enumerate(self.pages):
            sheet.paste(page.resize((thumb_w, thumb_h)), ((index % 3) * thumb_w, (index // 3) * thumb_h))
        sheet.save(self.directory / 'contact_sheet.png')
        return dict(slides=len(self.pages), pptx=str(self.directory / f'{stem}.pptx'), pdf=str(self.directory / f'{stem}.pdf'))
