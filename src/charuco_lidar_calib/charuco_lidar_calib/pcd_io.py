"""Minimal PCD reader (ASCII + binary), no PCL / open3d dependency.

Returns an (N, C) float32 array with the fields present in the header. Helper
``load_xyzi`` returns just (N, 4) = x, y, z, intensity.
"""
import numpy as np


_TYPE_MAP = {
    ('F', 4): np.float32, ('F', 8): np.float64,
    ('U', 1): np.uint8, ('U', 2): np.uint16, ('U', 4): np.uint32,
    ('I', 1): np.int8, ('I', 2): np.int16, ('I', 4): np.int32,
}


def read_pcd(path):
    """Return (fields, data) where data is an (N, len(fields)) float64 array."""
    with open(path, 'rb') as f:
        raw = f.read()
    # header ends right after the DATA line
    header_end = raw.find(b'\n', raw.find(b'DATA')) + 1
    header = raw[:header_end].decode('ascii', errors='replace')
    body = raw[header_end:]

    hdr = {}
    for line in header.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        hdr[parts[0].upper()] = parts[1:]

    fields = hdr['FIELDS']
    sizes = [int(x) for x in hdr['SIZE']]
    types = hdr['TYPE']
    counts = [int(x) for x in hdr.get('COUNT', ['1'] * len(fields))]
    npts = int(hdr['POINTS'][0]) if 'POINTS' in hdr else int(hdr['WIDTH'][0]) * int(hdr['HEIGHT'][0])
    data_kind = hdr['DATA'][0].lower()
    if data_kind not in ('ascii', 'binary'):
        raise ValueError(f"unsupported PCD DATA kind '{data_kind}' in {path} "
                         f"(binary_compressed is not supported)")

    # expand fields by count
    exp_fields, exp_dtypes = [], []
    for fld, sz, tp, cnt in zip(fields, sizes, types, counts):
        for c in range(cnt):
            name = fld if cnt == 1 else f'{fld}_{c}'
            exp_fields.append(name)
            exp_dtypes.append(_TYPE_MAP.get((tp, sz), np.float32))

    if data_kind == 'ascii':
        text = body.decode('ascii', errors='replace')
        arr = np.array(
            [[float(x) for x in ln.split()] for ln in text.splitlines() if ln.strip()],
            dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, len(exp_fields))
        return exp_fields, arr

    # binary (not binary_compressed)
    dt = np.dtype([(n, d) for n, d in zip(exp_fields, exp_dtypes)])
    structured = np.frombuffer(body[:npts * dt.itemsize], dtype=dt)
    out = np.zeros((len(structured), len(exp_fields)), dtype=np.float64)
    for i, n in enumerate(exp_fields):
        out[:, i] = structured[n].astype(np.float64)
    return exp_fields, out


def load_xyzi(path):
    """Return (N, 4) float64: x, y, z, intensity (intensity=0 if absent)."""
    fields, data = read_pcd(path)
    idx = {f.lower(): i for i, f in enumerate(fields)}
    x = data[:, idx['x']]
    y = data[:, idx['y']]
    z = data[:, idx['z']]
    inten = data[:, idx['intensity']] if 'intensity' in idx else np.zeros(len(x))
    out = np.column_stack([x, y, z, inten])
    # drop NaN/inf rows
    m = np.isfinite(out).all(axis=1)
    return out[m]
