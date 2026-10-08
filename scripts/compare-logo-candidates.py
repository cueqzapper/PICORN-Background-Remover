"""Apply the SAME white-graphics refinement to every candidate logo mask.
Run after benchmark-small-models.py. No tuning against the test fixtures.
"""
import json
from pathlib import Path
import runpy
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
bench = runpy.run_path(str(ROOT / 'scripts/benchmark-small-models.py'))
OUT = ROOT / '.verification/small-models'
WORK = OUT / 'logo-work'
WORK.mkdir(exist_ok=True)
rows, pictures = [], []
sessions = {name: bench['session'](name) for name in ['picorn512', 'u2netp', 'sinet-fixed']}
for fixture in ['navy-wordmark', 'pale-wordmark', 'white-artwork']:
    folder = ROOT / '.verification/white-graphics' / fixture
    source = Image.open(folder / 'source.png').convert('RGB')
    truth = np.asarray(Image.open(folder / 'ground-truth.png').getchannel('A')).astype(np.float32) / 255
    np.asarray(source.convert('RGBA')).tofile(WORK / 'source.rgba')
    picture = Image.new('RGB', (1280, 180), 'white')
    for index, (name, sess) in enumerate(sessions.items()):
        data, geometry = bench['prepare'](name, source)
        raw = sess.run(None, bench['feed'](sess, data))[0]
        if name == 'picorn512':
            data['image'].tofile(WORK / 'image.f32'); raw.tofile(WORK / 'model.f32')
            subprocess.run(['node', str(ROOT / 'scripts/evaluate-white-graphics.mjs'), 'hybrid', str(WORK)], check=True)
            raw = np.fromfile(WORK / 'hybrid.f32', np.float32).reshape(1, 1, 512, 512)
        baseline = bench['finish'](name, raw, geometry, source.size)
        baseline.tofile(WORK / 'baseline.f32')
        subprocess.run(['node', str(ROOT / 'scripts/evaluate-white-graphics.mjs'), 'graphic', str(WORK), str(source.width), str(source.height)], check=True)
        info = json.loads((WORK / 'result.json').read_text())
        alpha = np.fromfile(WORK / 'improved.f32', np.float32).reshape(source.height, source.width) if info['accepted'] else baseline
        foreground = np.fromfile(WORK / 'foreground.rgba', np.uint8).reshape(source.height, source.width, 4)[:, :, :3] if info['accepted'] else np.asarray(source)
        rows.append({'fixture': fixture, 'model': name, **info, 'quality': bench['score'](alpha, truth)})
        rgb = foreground * alpha[:, :, None] + np.array([65, 69, 78]) * (1 - alpha[:, :, None])
        panel = Image.fromarray(np.rint(rgb).clip(0, 255).astype(np.uint8)).resize((320, 160))
        picture.paste(panel, (320 * (index + 1), 20))
        ImageDraw.Draw(picture).text((320 * (index + 1) + 4, 4), name + ' + SAME graphic stage', fill='black')
    picture.paste(source.resize((320, 160)), (0, 20))
    ImageDraw.Draw(picture).text((4, 4), fixture, fill='black')
    pictures.append(picture)
sheet = Image.new('RGB', (1280, 540), 'white')
for i, row in enumerate(pictures): sheet.paste(row, (0, i * 180))
sheet.save(OUT / 'logo-pipelines.png')
(OUT / 'logo-pipelines.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
