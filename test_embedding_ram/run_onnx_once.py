#!/usr/bin/env python3
import argparse
import json
import urllib.request
from io import BytesIO

import numpy as np
import onnxruntime as ort


def _default_dim(d, fallback):
    if isinstance(d, int) and d > 0:
        return d
    return fallback


def _dtype_from_onnx(onnx_type):
    t = onnx_type.lower()
    if "float16" in t:
        return np.float16
    if "float" in t:
        return np.float32
    if "double" in t:
        return np.float64
    if "int8" in t:
        return np.int8
    if "uint8" in t:
        return np.uint8
    if "int16" in t:
        return np.int16
    if "uint16" in t:
        return np.uint16
    if "int32" in t:
        return np.int32
    if "uint32" in t:
        return np.uint32
    if "int64" in t:
        return np.int64
    if "bool" in t:
        return np.bool_
    return np.float32


def _safe_numeric_summary(arr):
    arr_np = np.asarray(arr)
    flat = arr_np.reshape(-1)
    sample = flat[:5]

    if arr_np.dtype == np.bool_:
        sample_list = [bool(x) for x in sample]
        return {
            "dtype": str(arr_np.dtype),
            "shape": list(arr_np.shape),
            "sample_first5": sample_list,
        }

    arr_f = arr_np.astype(np.float64, copy=False)
    sample_list = [float(x) for x in sample]
    return {
        "dtype": str(arr_np.dtype),
        "shape": list(arr_np.shape),
        "min": float(np.min(arr_f)) if arr_f.size else None,
        "max": float(np.max(arr_f)) if arr_f.size else None,
        "mean": float(np.mean(arr_f)) if arr_f.size else None,
        "sample_first5": sample_list,
    }


def _tokenize_ascii(query, seq_len, vocab_mod=32000):
    byte_vals = [int(b) for b in query.encode("utf-8", errors="ignore")]
    if not byte_vals:
        byte_vals = [0]

    ids = np.zeros((1, seq_len), dtype=np.int64)
    usable = min(seq_len, len(byte_vals))
    for i in range(usable):
        ids[0, i] = byte_vals[i] % vocab_mod

    attn = np.zeros((1, seq_len), dtype=np.int64)
    attn[0, :usable] = 1
    return ids, attn


def _load_lorem_picsum_rgb(target_h, target_w):
    # Use picsum source image to satisfy image-input requirement.
    url = "https://picsum.photos/512"
    with urllib.request.urlopen(url, timeout=20) as resp:
        img_bytes = resp.read()

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for image decode. Install with: pip install pillow") from exc

    im = Image.open(BytesIO(img_bytes)).convert("RGB")
    im = im.resize((target_w, target_h), Image.BILINEAR)
    arr = np.asarray(im)
    return arr


def _build_image_tensor(h, w, dtype):
    arr_hwc = _load_lorem_picsum_rgb(h, w)

    if np.issubdtype(dtype, np.floating):
        arr = (arr_hwc.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        return arr.astype(dtype)

    if np.issubdtype(dtype, np.integer):
        arr = arr_hwc.transpose(2, 0, 1)[None, ...]
        info = np.iinfo(dtype)
        return np.clip(arr, info.min, info.max).astype(dtype)

    arr = (arr_hwc.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return arr.astype(np.float32)


def _make_input_tensor(inp, query):
    name = inp.name.lower()
    shape = list(inp.shape)
    onnx_type = inp.type
    dtype = _dtype_from_onnx(onnx_type)

    is_text_ids = "input_ids" in name or "token" in name
    is_attn = "attention" in name and "mask" in name
    is_image = any(k in name for k in ["pixel", "image", "vision"])

    rank = len(shape)

    if is_text_ids:
        seq_len = _default_dim(shape[-1] if rank >= 2 else None, 16)
        ids, _ = _tokenize_ascii(query, seq_len)
        return ids.astype(dtype)

    if is_attn:
        seq_len = _default_dim(shape[-1] if rank >= 2 else None, 16)
        _, attn = _tokenize_ascii(query, seq_len)
        return attn.astype(dtype)

    if is_image or (rank == 4 and _default_dim(shape[1], 3) in (1, 3, 4)):
        b = _default_dim(shape[0], 1)
        c = _default_dim(shape[1], 3)
        h = _default_dim(shape[2], 224)
        w = _default_dim(shape[3], 224)
        img = _build_image_tensor(h, w, dtype)
        if b != 1:
            img = np.repeat(img, b, axis=0)
        if c != img.shape[1]:
            if c == 1:
                img = np.mean(img, axis=1, keepdims=True).astype(img.dtype)
            elif c > img.shape[1]:
                reps = int(np.ceil(c / img.shape[1]))
                img = np.tile(img, (1, reps, 1, 1))[:, :c, :, :]
            else:
                img = img[:, :c, :, :]
        return img

    final_shape = []
    for i, d in enumerate(shape):
        if i == 0:
            final_shape.append(_default_dim(d, 1))
        else:
            final_shape.append(_default_dim(d, 16 if i == rank - 1 else 1))

    if np.issubdtype(dtype, np.integer):
        return np.zeros(final_shape, dtype=dtype)
    if np.issubdtype(dtype, np.bool_):
        return np.zeros(final_shape, dtype=np.bool_)
    return np.zeros(final_shape, dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--query", default="a photo of a golden retriever in sunlight")
    args = parser.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    feed = {}
    input_meta = []
    for inp in sess.get_inputs():
        arr = _make_input_tensor(inp, args.query)
        feed[inp.name] = arr
        input_meta.append(
            {
                "name": inp.name,
                "onnx_type": inp.type,
                "declared_shape": [str(x) for x in inp.shape],
                "fed_shape": list(arr.shape),
                "fed_dtype": str(arr.dtype),
            }
        )

    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(output_names, feed)

    output_meta = {}
    for name, out in zip(output_names, outputs):
        output_meta[name] = _safe_numeric_summary(out)

    print(
        json.dumps(
            {
                "model": args.model,
                "query": args.query,
                "inputs": input_meta,
                "outputs": output_meta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
