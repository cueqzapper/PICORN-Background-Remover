"""Compare actual ONNX models on CPU, without changing production weights.

Run after downloading U2NETP and SINet into .verification/small-models.
Dependencies: numpy, Pillow, opencv-python, onnx, onnxruntime.
The SINet preprocessing follows its publisher's SINet_ONNX.ipynb (BGR,
dataset mean/std, then /255); U2NETP follows rembg's U2netpSession.
"""
import collections
import hashlib
import json
import platform
from pathlib import Path
import random
import subprocess
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.verification/small-models'
DATA = Path('G:/SEEZ/IMAGE-SEGMENTATION/training/data')
WORK = OUT / 'work'
WORK.mkdir(parents=True, exist_ok=True)
MODEL_PATHS = {
    'picorn512': ROOT / 'public/models/picorn-remove-background-v9.onnx',
    'picorn320': ROOT / 'public/models/picorn-remove-background-v9.onnx',
    'u2netp': OUT / 'u2netp.onnx',
    'sinet': OUT / 'sinet.onnx',
}
assert hashlib.md5(MODEL_PATHS['u2netp'].read_bytes()).hexdigest() == '8e83ca70e441ab06c318d82300c84806'
# Old SINet export lists fixed weights as overridable inputs. Remove only those
# input declarations so ORT can fold constants; keep an untouched original.
graph = onnx.load(MODEL_PATHS['sinet'])
weights = {x.name for x in graph.graph.initializer}
inputs = [x for x in graph.graph.input if x.name not in weights]
del graph.graph.input[:]
graph.graph.input.extend(inputs)
onnx.checker.check_model(graph)
optimized = OUT / 'sinet-fixed-inputs.onnx'
onnx.save(graph, optimized)
MODEL_PATHS['sinet-fixed'] = optimized


def session(name, threads=1):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(MODEL_PATHS[name]), sess_options=opts, providers=['CPUExecutionProvider'])


def prepare(name, image):
    if name.startswith('picorn'):
        size = int(name[6:])
        scale = size / max(image.size)
        w, h = [max(1, int(x * scale + 0.5)) for x in image.size]
        left, top = (size - w) // 2, (size - h) // 2
        boxed = Image.new('RGB', (size, size), (114, 114, 114))
        boxed.paste(image.resize((w, h), Image.Resampling.BILINEAR), (left, top))
        tensor = np.asarray(boxed).astype(np.float32).transpose(2, 0, 1)[None] / 255
        zeros = np.zeros((1, 1, size, size), np.float32)
        return {'image': tensor, 'guidance': zeros, 'guidance_weight': np.zeros(1, np.float32),
                'keep_hint': zeros, 'remove_hint': zeros}, (left, top, w, h)
    if name == 'u2netp':
        rgb = np.asarray(image.resize((320, 320), Image.Resampling.LANCZOS)).astype(np.float64)
        rgb /= max(rgb.max(), 1e-6)
        rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    else:
        rgb = cv2.resize(np.asarray(image)[:, :, ::-1], (320, 320)).astype(np.float32)
        rgb = ((rgb - [102.890434, 111.25247, 126.91212]) / [62.93292, 62.82138, 66.355705]) / 255
    return rgb.transpose(2, 0, 1)[None].astype(np.float32), None


def feed(sess, prepared):
    return prepared if isinstance(prepared, dict) else {sess.get_inputs()[0].name: prepared}


def finish(name, output, geometry, dimensions):
    if name.startswith('picorn'):
        left, top, w, h = geometry
        matte = output.squeeze()[top:top+h, left:left+w]
    elif name == 'u2netp':
        matte = output[0, 0]
        matte = (matte - matte.min()) / max(float(matte.max() - matte.min()), 1e-8)
        # Reproduce rembg's 8-bit output before Lanczos resizing.
        return np.asarray(Image.fromarray((matte * 255).astype(np.uint8)).resize(dimensions, Image.Resampling.LANCZOS)).astype(np.float32) / 255
    else:
        matte = output[0, 1]
    return np.asarray(Image.fromarray(matte).resize(dimensions, Image.Resampling.BILINEAR)).clip(0, 1)


def score(prediction, truth):
    assert np.isfinite(prediction).all()
    pred, target = prediction >= 0.5, truth >= 0.5
    intersection, union = np.logical_and(pred, target).sum(), np.logical_or(pred, target).sum()
    kernel = np.ones((3, 3), np.uint8)
    pe = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    te = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    precision = (pe & cv2.dilate(te, kernel)).sum() / max(1, pe.sum())
    recall = (te & cv2.dilate(pe, kernel)).sum() / max(1, te.sum())
    return {'iou': float(intersection / max(1, union)), 'mae': float(np.abs(prediction - truth).mean()),
            'boundary_f1_1px': float(2 * precision * recall / max(1e-8, precision + recall))}


def main():
    groups = collections.defaultdict(list)
    for filename in ['hqseg44k-manifest.jsonl', 'p3m10k-manifest.jsonl']:
        for line in (DATA / filename).read_text().splitlines():
            record = json.loads(line)
            if record['split'] == 'val': groups[record['source']].append(record)
    records = []
    for source in ['DIS5K', 'DUTS-TE', 'P3M-10K']:
        candidates = sorted(groups[source], key=lambda r: r['image'])
        random.Random(20260909).shuffle(candidates)
        records.extend(candidates[:32])
    for name in ['navy-wordmark', 'pale-wordmark', 'white-artwork']:
        folder = ROOT / '.verification/white-graphics' / name
        records.append({'image': str(folder / 'source.png'), 'mask': str(folder / 'ground-truth.png'),
                        'source': 'synthetic-logos', 'split': 'diagnostic', 'alpha_channel': True})
    (OUT / 'selection.json').write_text(json.dumps(records, indent=2))
    sessions = {name: session(name) for name in MODEL_PATHS}
    sample = Image.open(records[0]['image']).convert('RGB')
    performance = {}
    for name in MODEL_PATHS:
        performance[name] = {'bytes': MODEL_PATHS[name].stat().st_size,
            'sha256': hashlib.sha256(MODEL_PATHS[name].read_bytes()).hexdigest()}
        for threads in [1, 4]:
            start = time.perf_counter(); sess = session(name, threads)
            load_ms = (time.perf_counter() - start) * 1000
            data, geometry = prepare(name, sample)
            feeds = feed(sess, data)
            start = time.perf_counter(); sess.run(None, feeds)
            first_ms = (time.perf_counter() - start) * 1000
            for _ in range(5): sess.run(None, feeds)
            times = []
            for _ in range(50):
                start = time.perf_counter(); sess.run(None, feeds)
                times.append((time.perf_counter() - start) * 1000)
            performance[name][f'cpu_threads_{threads}'] = {'session_load_ms': load_ms, 'first_run_ms': first_ms,
                'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)), 'repeats': 50}
        print(name, json.dumps(performance[name]), flush=True)
    rows, pictures = [], []
    parity = 0
    for index, record in enumerate(records):
        image = Image.open(record['image']).convert('RGB')
        truth_image = Image.open(record['mask'])
        truth_image = truth_image.getchannel('A') if record.get('alpha_channel') else truth_image.convert('L')
        # Fixed evaluation canvas, same for all models; preserve original aspect.
        size = image.size if record['source'] == 'synthetic-logos' else tuple(max(1, int(x * min(1, 512 / max(image.size)) + .5)) for x in image.size)
        truth = np.asarray(truth_image.resize(size, Image.Resampling.BILINEAR)).astype(np.float32) / 255
        predictions = {}
        raw_sinet = None
        for name, sess in sessions.items():
            data, geometry = prepare(name, image)
            start = time.perf_counter(); raw = sess.run(None, feed(sess, data))[0]
            elapsed = (time.perf_counter() - start) * 1000
            prediction = finish(name, raw, geometry, size)
            rows.append({'index': index, 'source': record['source'], 'model': name, 'model_ms': elapsed, **score(prediction, truth)})
            predictions[name] = prediction
            if name == 'sinet': raw_sinet = raw
            if name == 'sinet-fixed': parity = max(parity, float(np.abs(raw - raw_sinet).max()))
            if name == 'picorn512':
                data['image'].tofile(WORK / 'image.f32'); raw.tofile(WORK / 'model.f32')
                subprocess.run(['node', str(ROOT / 'scripts/evaluate-white-graphics.mjs'), 'hybrid', str(WORK)], check=True, capture_output=True)
                hybrid = np.fromfile(WORK / 'hybrid.f32', np.float32).reshape(1, 1, 512, 512)
                baseline = finish(name, hybrid, geometry, size)
                predictions['picorn-v10'] = baseline
                rows.append({'index': index, 'source': record['source'], 'model': 'picorn-v10', **score(baseline, truth)})
                if record['source'] == 'synthetic-logos':
                    np.asarray(image.convert('RGBA')).tofile(WORK / 'source.rgba'); baseline.tofile(WORK / 'baseline.f32')
                    subprocess.run(['node', str(ROOT / 'scripts/evaluate-white-graphics.mjs'), 'graphic', str(WORK), str(size[0]), str(size[1])], check=True, capture_output=True)
                    info = json.loads((WORK / 'result.json').read_text())
                    refined = np.fromfile(WORK / 'improved.f32', np.float32).reshape(size[1], size[0]) if info['accepted'] else baseline
                    predictions['picorn-graphic'] = refined
                    rows.append({'index': index, 'source': record['source'], 'model': 'picorn-graphic', **score(refined, truth)})
        # Predetermined first two in each group plus every synthetic logo.
        if index in [0, 1, 32, 33, 64, 65, 96, 97, 98]:
            names = ['truth', 'picorn-v10', 'u2netp', 'sinet-fixed']
            row = Image.new('RGB', (1200, 230), 'white')
            row.paste(ImageOps.contain(image, (240, 195)), (0, 30))
            ImageDraw.Draw(row).text((5, 8), f'{index}: {record["source"]}', fill='black')
            rgb = np.asarray(image.resize(size)).astype(np.float32)
            for col, name in enumerate(names, 1):
                alpha = truth if name == 'truth' else predictions[name]
                composite = np.rint(rgb * alpha[:, :, None] + np.array([65, 69, 78]) * (1 - alpha[:, :, None])).clip(0,255).astype(np.uint8)
                row.paste(ImageOps.contain(Image.fromarray(composite), (240, 195)), (col * 240, 30))
                ImageDraw.Draw(row).text((col * 240 + 5, 8), name, fill='black')
            pictures.append(row)
        if (index + 1) % 16 == 0: print(f'Evaluated {index + 1}/{len(records)} images', flush=True)
    summary = {}
    for source in ['DIS5K', 'DUTS-TE', 'P3M-10K', 'synthetic-logos']:
        summary[source] = {}
        for name in [*MODEL_PATHS, 'picorn-v10', 'picorn-graphic']:
            selected = [r for r in rows if r['source'] == source and r['model'] == name]
            if selected: summary[source][name] = {'n': len(selected), **{metric: float(np.mean([r[metric] for r in selected])) for metric in ['iou', 'mae', 'boundary_f1_1px']}}
    sheet = Image.new('RGB', (1200, 230 * len(pictures)), 'white')
    for n, pic in enumerate(pictures): sheet.paste(pic, (0, n * 230))
    sheet.save(OUT / 'comparison.png')
    result = {'environment': {'platform': platform.platform(), 'processor': platform.processor(), 'onnxruntime': ort.__version__,
                             'provider': 'CPUExecutionProvider', 'quality_threads': 1, 'seed': 20260909},
              'performance': performance, 'quality': summary, 'sinet_optimization_max_absolute_error': parity, 'per_image': rows}
    (OUT / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'quality': summary, 'sinet_optimization_max_absolute_error': parity}, indent=2), flush=True)


if __name__ == '__main__': main()
