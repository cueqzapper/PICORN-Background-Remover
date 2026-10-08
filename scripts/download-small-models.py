"""Download only the two benchmark candidates; verify the exact tested bytes."""
import hashlib
from pathlib import Path
import urllib.request

OUT = Path(__file__).resolve().parents[1] / '.verification/small-models'
OUT.mkdir(parents=True, exist_ok=True)
CANDIDATES = {
    'u2netp.onnx': (
        'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx',
        '309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8',
    ),
    'sinet.onnx': (
        'https://raw.githubusercontent.com/anilsathyan7/Portrait-Segmentation/dbf69b043cf70d3362bc500ee620f20807e622d2/SINet/SINet_Softmax.onnx',
        '932c83e0013e07bbe597fe0007ba5ae6add3fbb6413ca5b81f18b2d410ac7259',
    ),
}
for name, (url, expected) in CANDIDATES.items():
    destination = OUT / name
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError(f'Unexpected downloaded model hash: {name}')
        destination.write_bytes(payload)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Unexpected existing model hash: {name}')
    print(f'{name}: verified {destination.stat().st_size} bytes')
