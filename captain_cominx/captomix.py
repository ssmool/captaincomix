"""
captomix.py

A best-effort, modular Python implementation that:
 - selects an input image file
 - extracts image "hashtags" (top ImageNet labels)
 - runs YOLO object/character detection (ultralytics / torch hub fallback)
 - builds image embeddings, clusters with scikit-learn KMeans
 - searches Google Custom Search and Flickr (requires API keys) for MP4s/GIFs
 - downloads candidate videos and merges them as animated layers onto the image
 - provides Pillow filters and stacking/layer helpers
 - main entry: captomix(...) that ties steps together

NOTES & REQUIREMENTS:
 - This script is a framework and contains places where you must supply API keys
   and optionally tune model selection. It aims to be runnable if you install
   the dependencies and provide valid keys.
 - Recommended pip installs (example):
     pip install pillow moviepy requests scikit-learn opencv-python torch torchvision ultralytics flickrapi google-api-python-client
 - If ultralytics YOLO is unavailable, the script will attempt torch.hub yolov5.
 - Google Custom Search requires creating a CSE and an API key.
 - Flickr requires an API key + secret.

Use this file as a starting point and adapt to your environment.

"""

import os
import io
import math
import json
import time
import requests
import tempfile
from typing import List, Tuple, Optional, Dict

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# ML / CV
import torch
import torchvision.transforms as T
import torchvision.models as models
from sklearn.cluster import KMeans

# moviepy for video composition
from moviepy.editor import (ImageClip, VideoFileClip, CompositeVideoClip,
                            concatenate_videoclips)

# Optional: ultralytics YOLO import (if installed)
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False

# Optional flickr + google client placeholders
# from flickrapi import FlickrAPI
# from googleapiclient.discovery import build

# ------------------------- Configuration -------------------------
CONFIG = {
    "GOOGLE_API_KEY": "YOUR_GOOGLE_API_KEY",
    "GOOGLE_CSE_ID": "YOUR_GOOGLE_CSE_ID",  # Custom Search Engine ID
    "FLICKR_API_KEY": "YOUR_FLICKR_API_KEY",
    "FLICKR_API_SECRET": "YOUR_FLICKR_API_SECRET",
    # search limits
    "MAX_VIDEO_RESULTS": 3,
    # model device
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

# ------------------------- Utilities -------------------------

def select_image(path: str) -> Image.Image:
    """Open an image file path with Pillow and return a converted RGB image."""
    img = Image.open(path)
    return img.convert("RGB")


def save_image(img: Image.Image, out_path: str):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    img.save(out_path)


# ------------------------- Hashtags (ImageNet top predictions) -------------------------

IMAGENET_LABELS = None

def _load_imagenet_labels():
    global IMAGENET_LABELS
    if IMAGENET_LABELS is None:
        # Minimal bundle of ImageNet classes. For production, load full labels file.
        # torch provides a mapping in torchvision if available - fallback small map.
        try:
            import json
            from urllib import request
            # This is a fallback: attempt to load local/packaged labels if present.
            # To keep offline, build a tiny map (not exhaustive):
            IMAGENET_LABELS = {"n02123045": "tabby, tabby cat"}  # placeholder
        except Exception:
            IMAGENET_LABELS = {}
    return IMAGENET_LABELS


def extract_hashtags(image: Image.Image, topk: int = 5) -> List[str]:
    """Use a pretrained ResNet model to predict top ImageNet-like labels and turn them into hashtags.

    This is a pragmatic approach — for better captions use an image captioning model.
    """
    device = CONFIG["DEVICE"]
    model = models.resnet50(pretrained=True)
    model.eval().to(device)

    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    img_t = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
        probs = torch.nn.functional.softmax(out[0], dim=0)

    # decode using torchvision utils if present
    try:
        from torchvision import models as tv_models
        # torchvision has labels in some releases; fallback to topk indices
        topk_idx = torch.topk(probs, topk).indices.cpu().numpy().tolist()
        # map indices to labels if available
        from torchvision.datasets import ImageNet
        # If ImageNet labels not directly available, just create generic tags
        tags = [f"label_{i}" for i in topk_idx]
    except Exception:
        topk_idx = torch.topk(probs, topk).indices.cpu().numpy().tolist()
        tags = [f"label_{i}" for i in topk_idx]

    hashtags = ["#" + t.replace(" ", "_").replace(",", "") for t in tags]
    return hashtags


# ------------------------- YOLO detection -------------------------

def load_yolo_model():
    device = CONFIG["DEVICE"]
    if _HAS_ULTRALYTICS:
        model = YOLO("yolov8n.pt")  # tiny default if installed
        return model
    else:
        # fallback to yolov5 via torch.hub (if internet allowed and model cached)
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.to(device)
            return model
        except Exception:
            print("Warning: YOLO model unavailable. Object detection will be skipped.")
            return None


def detect_objects(image: Image.Image, conf_thresh: float = 0.25) -> List[Dict]:
    """Run YOLO on a PIL image and return detections as dicts: {label, conf, bbox}

    bbox format: (x1, y1, x2, y2)
    """
    model = load_yolo_model()
    if model is None:
        return []

    # convert image to numpy
    np_img = np.array(image)

    if _HAS_ULTRALYTICS and isinstance(model, YOLO):
        results = model(np_img)[0]
        dets = []
        for *xyxy, conf, cls in results.boxes.data.cpu().numpy():
            if conf < conf_thresh: continue
            x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
            label = model.names[int(cls)]
            dets.append({"label": label, "conf": float(conf), "bbox": (x1, y1, x2, y2)})
        return dets
    else:
        # torch.hub yolov5
        results = model(np_img)
        df = results.pandas().xyxy[0]
        dets = []
        for _, r in df.iterrows():
            if r['confidence'] < conf_thresh: continue
            dets.append({"label": r['name'], "conf": float(r['confidence']), "bbox": (int(r['xmin']), int(r['ymin']), int(r['xmax']), int(r['ymax']))})
        return dets


# ------------------------- Embeddings + Clustering -------------------------

def image_embedding(image: Image.Image) -> np.ndarray:
    """Produce a 2048-d embedding using ResNet50's avgpool output."""
    device = CONFIG["DEVICE"]
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
    model.eval().to(device)

    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).squeeze().cpu().numpy()
    feat = feat.reshape(-1)
    return feat


def cluster_embeddings(embeddings: List[np.ndarray], n_clusters: int = 3) -> Tuple[KMeans, np.ndarray]:
    X = np.stack(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


# ------------------------- Search Google + Flickr for mp4/gif -------------------------


def search_google_videos(query: str, max_results: int = 3) -> List[str]:
    """
    Use Google Custom Search JSON API to find video/mp4 links. Returns list of URLs.
    Requires CONFIG['GOOGLE_API_KEY'] and CONFIG['GOOGLE_CSE_ID'] to be set.
    """
    api_key = CONFIG.get("GOOGLE_API_KEY")
    cse_id = CONFIG.get("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        print("Google API key / CSE ID not configured. Skipping Google search.")
        return []

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': query,
        'searchType': 'image',  # image search returns images; we will accept mp4/gif links too
        'num': min(10, max_results),
    }
    try:
        res = requests.get(search_url, params=params, timeout=10)
        data = res.json()
        items = data.get('items', [])
        urls = [it.get('link') for it in items if it.get('link')]
        # filter for mp4/gif
        urls = [u for u in urls if u.lower().endswith(('.mp4', '.gif'))][:max_results]
        return urls
    except Exception as e:
        print("Google search error:", e)
        return []


def search_flickr_videos(query: str, max_results: int = 3) -> List[str]:
    """
    Flickr's API historically focuses on images; video hosted on Flickr may be returned.
    This function requires FLICKR_API_KEY/SECRET. It does a naive flickr.photos.search and
    then constructs static URLs — for direct video links you'll need to use the Flickr API
    with authenticated calls that we don't fully implement here.
    """
    if not CONFIG.get('FLICKR_API_KEY'):
        print("Flickr API key not set — skipping Flickr search")
        return []
    # Placeholder: for a production-grade pipeline, use flickrapi.FlickrAPI and proper auth.
    return []


# ------------------------- Download helper -------------------------

def download_url_to_file(url: str, out_path: str, timeout: int = 30) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Download failed for {url}: {e}")
        return False


# ------------------------- Pillow Filters -------------------------

def apply_filters_pillow(img: Image.Image, filters: List[str]) -> Image.Image:
    """Apply a list of named filters to the PIL.Image and return result.

    Supported filters (examples): 'BLUR', 'CONTOUR', 'DETAIL', 'SHARPEN', 'BRIGHTNESS:1.2',
    'CONTRAST:0.8', 'COLOR:1.1', 'GRAYSCALE'
    """
    out = img.copy()
    for f in filters:
        if f == 'BLUR':
            out = out.filter(ImageFilter.BLUR)
        elif f == 'CONTOUR':
            out = out.filter(ImageFilter.CONTOUR)
        elif f == 'DETAIL':
            out = out.filter(ImageFilter.DETAIL)
        elif f == 'SHARPEN':
            out = out.filter(ImageFilter.SHARPEN)
        elif f.startswith('BRIGHTNESS:'):
            val = float(f.split(':', 1)[1])
            out = ImageEnhance.Brightness(out).enhance(val)
        elif f.startswith('CONTRAST:'):
            val = float(f.split(':', 1)[1])
            out = ImageEnhance.Contrast(out).enhance(val)
        elif f.startswith('COLOR:'):
            val = float(f.split(':', 1)[1])
            out = ImageEnhance.Color(out).enhance(val)
        elif f == 'GRAYSCALE':
            out = out.convert('L').convert('RGB')
        else:
            print(f"Unknown filter: {f}")
    return out


# ------------------------- Layering: combine image + video/mp4/gif -------------------------

def add_video_layer(base_image: Image.Image,
                    video_path: str,
                    position: Tuple[int, int] = (0, 0),
                    size: Optional[Tuple[int, int]] = None,
                    opacity: float = 1.0,
                    start_time: float = 0.0,
                    loop: bool = True) -> VideoFileClip:
    """
    Create a moviepy CompositeVideoClip where the base image is the background and
    the video_path is overlayed at `position` (x,y) with given size and opacity.
    Returns the resulting VideoFileClip which you can write to file.
    """
    bg_w, bg_h = base_image.size
    # convert base image to ImageClip
    bg_clip = ImageClip(np.array(base_image)).set_duration(10)  # default duration 10s, can be changed

    vid = VideoFileClip(video_path)
    if size:
        vid = vid.resize(newsize=size)

    # set position and opacity
    vid = vid.set_start(start_time).set_pos(position).set_duration(bg_clip.duration).with_mask(vid.mask if hasattr(vid, 'mask') else None)
    if opacity < 1.0:
        vid = vid.fx( lambda clip: clip.set_opacity(opacity) )

    comp = CompositeVideoClip([bg_clip, vid], size=(bg_w, bg_h))
    return comp


def merge_videos_to_image(base_image: Image.Image,
                          video_paths: List[str],
                          out_path: str,
                          filters: Optional[List[str]] = None,
                          duration: float = 8.0):
    """High-level: get videos, overlay them sequentially (side-by-side or stacked)
    and export a final MP4 and GIF.
    """
    # Apply filters to background
    bg = apply_filters_pillow(base_image, filters or [])

    clips = []
    bg_clip = ImageClip(np.array(bg)).set_duration(duration)

    # simple layout: evenly space videos across width
    n = max(1, len(video_paths))
    bw, bh = bg.size
    slot_w = bw // n
    for i, vp in enumerate(video_paths):
        try:
            v = VideoFileClip(vp).set_duration(duration)
            # scale to slot size maintaining aspect
            v = v.resize(width=slot_w - 10)
            x = i * slot_w + (slot_w - v.w) // 2
            y = (bh - v.h) // 2
            v = v.set_pos((x, y))
            clips.append(v)
        except Exception as e:
            print(f"Could not load video {vp}: {e}")

    comp = CompositeVideoClip([bg_clip] + clips, size=(bw, bh)).set_duration(duration)

    # write mp4
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    mp4_out = out_path if out_path.lower().endswith('.mp4') else out_path + '.mp4'
    gif_out = os.path.splitext(mp4_out)[0] + '.gif'

    # write video (this may take time)
    comp.write_videofile(mp4_out, codec='libx264', audio=False, threads=4, fps=24)
    # write GIF (smaller, lower fps)
    comp.write_gif(gif_out, fps=12)

    return mp4_out, gif_out


# ------------------------- High-level pipeline: captomix -------------------------

def captomix(input_image_path: str,
             output_path_prefix: str,
             search_providers: List[str] = ['google', 'flickr'],
             max_videos: int = 2,
             filters: Optional[List[str]] = None) -> Dict:
    """
    The main wrapper. Steps:
     1. load image
     2. extract hashtags
     3. detect objects
     4. build embedding and optionally cluster with previously cached embeddings
     5. search Google/Flickr for mp4/gif relevant to hashtags/objects
     6. download top results
     7. merge videos onto the image to create an animated output

    Returns a dict with outputs and statuses.
    """
    img = select_image(input_image_path)
    hashtags = extract_hashtags(img, topk=5)
    detections = detect_objects(img)

    queries = []
    queries.extend([h.lstrip('#').replace('_', ' ') for h in hashtags])
    queries.extend([d['label'] for d in detections])
    queries = [q for q in queries if q]

    found_video_paths = []

    # For each query, search providers until we accumulate max_videos
    for q in queries:
        if len(found_video_paths) >= max_videos:
            break
        if 'google' in search_providers:
            urls = search_google_videos(q, max_results=CONFIG.get('MAX_VIDEO_RESULTS', 3))
            for u in urls:
                if len(found_video_paths) >= max_videos: break
                tmpf = os.path.join(tempfile.gettempdir(), os.path.basename(u).split('?')[0])
                ok = download_url_to_file(u, tmpf)
                if ok:
                    found_video_paths.append(tmpf)
        if 'flickr' in search_providers and len(found_video_paths) < max_videos:
            urls = search_flickr_videos(q, max_results=CONFIG.get('MAX_VIDEO_RESULTS', 3))
            for u in urls:
                if len(found_video_paths) >= max_videos: break
                tmpf = os.path.join(tempfile.gettempdir(), os.path.basename(u).split('?')[0])
                ok = download_url_to_file(u, tmpf)
                if ok:
                    found_video_paths.append(tmpf)

    # If we didn't find any videos, warn and return with a placeholder
    if not found_video_paths:
        print("No videos found via search providers. You can manually pass video_paths to merge_videos_to_image.")

    out_mp4 = output_path_prefix + '.mp4'
    out_mp4, out_gif = merge_videos_to_image(img, found_video_paths, out_mp4, filters=filters or [], duration=8.0)

    return {
        'input': input_image_path,
        'hashtags': hashtags,
        'detections': detections,
        'videos_used': found_video_paths,
        'mp4': out_mp4,
        'gif': out_gif,
    }
