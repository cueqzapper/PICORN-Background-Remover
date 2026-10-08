"""Reproducible synthetic logo evaluation with the actual shipped ONNX model.
Requires numpy, Pillow and onnxruntime. Saves originals, alpha truth, outputs,
comparison sheet and metrics. These are diagnostics, not a customer benchmark.
Run: python scripts/evaluate-white-graphics.py
"""
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.verification' / 'white-graphics'
OUT.mkdir(parents=True, exist_ok=True)
session = ort.InferenceSession(str(ROOT / 'public/models/picorn-remove-background-v9.onnx'), providers=['CPUExecutionProvider'])
font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 180)
rows, metrics = [], []
for name, color in [('navy-wordmark', (7, 62, 96)), ('pale-wordmark', (225, 225, 225)), ('white-artwork', (7, 62, 96))]:
    width, height = 1024, 512
    layer = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(layer)
    if name == 'white-artwork':
        draw.rounded_rectangle((100, 80, 450, 430), radius=35, fill=(*color, 255))
        draw.polygon([(275, 130), (310, 235), (420, 235), (332, 300), (366, 405), (275, 340), (184, 405), (218, 300), (130, 235), (240, 235)], fill='white')
        draw.text((510, 175), 'GO', font=font, fill=(*color, 255))
    else:
        draw.text((95, 155), 'PICORN', font=font, fill=(*color, 255))
        draw.line((100, 380, 900, 380), fill=(*color, 255), width=2)
    truth = np.asarray(layer)[:, :, 3].astype(np.float32) / 255
    source = Image.alpha_composite(Image.new('RGBA', layer.size, 'white'), layer)
    folder = OUT / name; folder.mkdir(exist_ok=True)
    source.save(folder / 'source.png'); layer.save(folder / 'ground-truth.png')
    np.asarray(source).tofile(folder / 'source.rgba')
    boxed = Image.new('RGB', (512, 512), (114, 114, 114))
    boxed.paste(source.convert('RGB').resize((512, 256), Image.Resampling.BILINEAR), (0, 128))
    image = np.asarray(boxed).astype(np.float32).transpose(2, 0, 1)[None] / 255
    zeros = np.zeros((1, 1, 512, 512), np.float32)
    feeds = {'image': image, 'guidance': zeros, 'guidance_weight': np.zeros(1, np.float32), 'keep_hint': zeros, 'remove_hint': zeros}
    start = time.perf_counter(); model = session.run(['alpha'], feeds)[0]
    model_ms = (time.perf_counter() - start) * 1000
    image.tofile(folder / 'image.f32'); model.tofile(folder / 'model.f32')
    subprocess.run(['node', str(ROOT / 'scripts/evaluate-white-graphics.mjs'), 'hybrid', str(folder)], check=True)
    hybrid = np.fromfile(folder / 'hybrid.f32', np.float32).reshape(512, 512)[128:384]
    baseline = np.asarray(Image.fromarray(hybrid).resize((width, height), Image.Resampling.BILINEAR))
    baseline.tofile(folder / 'baseline.f32')
    subprocess.run(['node', str(ROOT / 'scripts/evaluate-white-graphics.mjs'), 'graphic', str(folder), str(width), str(height)], check=True)
    info = json.loads((folder / 'result.json').read_text())
    improved = np.fromfile(folder / 'improved.f32', np.float32).reshape(height, width) if info['accepted'] else baseline
    foreground = np.fromfile(folder / 'foreground.rgba', np.uint8).reshape(height, width, 4) if info['accepted'] else np.asarray(source)
    def score(alpha):
        prediction, target = alpha >= 0.5, truth >= 0.5
        return {'iou': float(np.logical_and(prediction, target).sum() / max(1, np.logical_or(prediction, target).sum())), 'mae': float(np.abs(alpha - truth).mean())}
    metrics.append({'name': name, 'model_ms': model_ms, **info, 'baseline': score(baseline), 'improved': score(improved)})
    panels = [source.convert('RGB')]
    for title, rgb, alpha in [('baseline', np.asarray(source), baseline), ('improved', foreground, improved), ('truth', np.asarray(layer), truth)]:
        rgba = rgb.copy(); rgba[:, :, 3] = np.rint(alpha * 255).clip(0, 255).astype(np.uint8)
        cutout = Image.fromarray(rgba); cutout.save(folder / f'{title}.png')
        panel = Image.alpha_composite(Image.new('RGBA', layer.size, (60, 65, 75, 255)), cutout).convert('RGB')
        panels.append(panel)
    row = Image.new('RGB', (1024, 160), 'white')
    for index, panel in enumerate(panels):
        row.paste(panel.resize((256, 128)), (index * 256, 24))
        ImageDraw.Draw(row).text((index * 256 + 5, 5), [name, 'Before: V9 + V10', 'After: graphic masking', 'Ground truth'][index], fill='black')
    rows.append(row)
sheet = Image.new('RGB', (1024, len(rows) * 160), 'white')
for index, row in enumerate(rows): sheet.paste(row, (0, index * 160))
sheet.save(OUT / 'comparison.png')
(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
