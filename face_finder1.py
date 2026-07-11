from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Callable
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    raise SystemExit(
        "Failed to import insightface. Install with: pip install insightface onnxruntime"
    ) from e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".tiff", ".tif", ".dng"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def read_image_bgr(path: Path) -> np.ndarray:
    # Attempt to read with OpenCV first (supports JPG, PNG, etc.)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        ext = path.suffix.lower()
        # Fallback for RAW files (DNG, etc.)
        if ext in {".dng", ".tiff", ".tif"}:
            try:
                import rawpy
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess()
                    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                pass # Try PIL if rawpy fails

        # Fallback to PIL for HEIC and others
        if img is None:
            try:
                from PIL import Image
                import pillow_heif
                pillow_heif.register_heif_opener()
                pil_img = Image.open(path)
                # Convert PIL RGB to OpenCV BGR
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"Could not read image {path}: {e}")
            
    if img is None:
        raise ValueError(f"Could not read image: {path} (Unknown format/Corrupted)")
    return img


def save_image_bgr(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    
    # OpenCV cannot encode HEIC or RAW formats. Force JPG for these result images.
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        path = path.with_suffix(".jpg")
        ext = ".jpg"
        
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise ValueError(f"Could not encode image: {path}")
    path.write_bytes(buf.tobytes())


def collect_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and is_image(p)]


def load_face_app(det_size: int = 640, use_gpu: bool = True):
    providers = ["CPUExecutionProvider"]
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider"] + providers
    
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def build_known_embeddings(app: FaceAnalysis, known_dir: Path, min_size: int, verbose: bool = False):
    people = {}
    entities = sorted(known_dir.iterdir())
    
    if verbose:
        print(f"  Scanning: {known_dir}")
        print(f"  Found {len(entities)} items: {[e.name for e in entities]}")
    
    for item in entities:
        embs = []
        person_name = ""
        
        if item.is_dir():
            person_name = item.name
            target_files = collect_files(item)
        elif item.is_file() and is_image(item):
            person_name = item.stem
            target_files = [item]
        else:
            if verbose:
                print(f"  Skipping: {item.name} (not a directory or image)")
            continue
        
        if verbose:
            print(f"  Processing '{person_name}': {len(target_files)} images")
            
        for img_path in target_files:
            try:
                img = read_image_bgr(img_path)
                faces = app.get(img)
                if not faces:
                    if verbose:
                        print(f"    No faces detected in: {img_path.name}")
                    continue
                f0 = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                x1, y1, x2, y2 = map(int, f0.bbox)
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    if verbose:
                        print(f"    Face too small ({x2-x1}x{y2-y1}) in: {img_path.name}")
                    continue
                emb = f0.embedding
                if emb is not None:
                    embs.append(emb.astype(np.float32))
                    if verbose:
                        print(f"    Face found in: {img_path.name}")
            except Exception as e:
                if verbose:
                    print(f"    Error reading {img_path.name}: {e}")
                continue
                
        if embs:
            mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            people[person_name] = mean_emb.astype(np.float32)
            
    return people


def annotate_image(img, faces_info):
    out = img.copy()
    for (x1, y1, x2, y2), name, score in faces_info:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_rect = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, y_rect), (x1 + tw + 8, y_rect + th + 8), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 4, y_rect + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


def process_single_photo(img_path, app, people_embs, sim_threshold, min_face):
    try:
        img = read_image_bgr(img_path)
    except:
        return None

    faces = app.get(img)
    face_infos = []
    matches_in_photo = []
    
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        if (x2 - x1) < min_face or (y2 - y1) < min_face:
            continue
        emb = f.embedding
        if emb is None:
            continue
        
        emb = emb.astype(np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n

        best_name = None
        best_score = -1.0
        for name, ref_emb in people_embs.items():
            score = cosine_similarity(emb, ref_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name and best_score >= sim_threshold:
            face_infos.append(((x1, y1, x2, y2), best_name, best_score))
            matches_in_photo.append((best_name, best_score))

    return {
        "path": img_path,
        "img": img,
        "face_infos": face_infos,
        "matches": matches_in_photo
    }


def process_event_photos(app, people_embs, photos_dir, out_dir, sim_threshold, min_face, progress_callback=None):
    annotated_dir = out_dir / "annotated"
    per_person_dir = out_dir / "per_person"
    reports_dir = out_dir / "reports"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    per_person_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {name: [] for name in people_embs.keys()}

    files = collect_files(photos_dir)
    total_files = len(files)
    
    for i, img_path in enumerate(tqdm(files, desc="Scanning photos")):
        if progress_callback:
            progress_callback(i + 1, total_files, str(img_path.name))
        
        result = process_single_photo(img_path, app, people_embs, sim_threshold, min_face)
        if not result or not result["matches"]:
            continue
            
        matches_in_photo = result["matches"]
        img = result["img"]
        face_infos = result["face_infos"]
        
        ann = annotate_image(img, face_infos)
        ann_path = annotated_dir / (img_path.stem + "_annotated" + img_path.suffix)
        save_image_bgr(ann_path, ann)

        for name, score in matches_in_photo:
            dst = per_person_dir / name / img_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(img_path, dst)
            except:
                pass

        rows.append({
            "photo": str(img_path),
            "annotated": str(ann_path.resolve()),
            "matches": ",".join(sorted({m[0] for m in matches_in_photo})),
            "scores": json.dumps({name: float(score) for name, score in matches_in_photo}),
        })
        for name, score in matches_in_photo:
            summary[name].append({"photo": str(img_path), "score": float(score)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(reports_dir / "matches.csv", index=False)
        (reports_dir / "matches.json").write_text(json.dumps({"matches": summary}, indent=2))
    return df


def main():
    parser = argparse.ArgumentParser(description="Face Finder: Match faces in event photos against known individuals.")
    parser.add_argument("--known", type=str, default="Known", help="Directory containing folders of known individuals")
    parser.add_argument("--photos", type=str, default="Event Photos", help="Directory containing event photos to scan")
    parser.add_argument("--output", type=str, default="output", help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.45, help="Cosine similarity threshold (default: 0.45)")
    parser.add_argument("--det-size", type=int, default=640, help="Detection size for the face model (default: 640)")
    parser.add_argument("--min-face", type=int, default=40, help="Minimum face size in pixels (default: 40)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU acceleration if available")

    args = parser.parse_args()

    known_dir = Path(args.known)
    photos_dir = Path(args.photos)
    out_dir = Path(args.output)

    if not known_dir.exists():
        print(f"Error: Known directory not found at {known_dir}")
        return
    if not photos_dir.exists():
        print(f"Error: Photos directory not found at {photos_dir}")
        return

    print("Loading face models...")
    try:
        app = load_face_app(det_size=args.det_size, use_gpu=args.use_gpu)
    except Exception as e:
        print(f"Error loading face models: {e}")
        return

    print("Building reference embeddings...")
    people_embs = build_known_embeddings(app, known_dir, min_size=args.min_face, verbose=True)
    if not people_embs:
        print("No valid reference faces found in the 'Known' directory folders!")
        return

    print(f"Loaded {len(people_embs)} individuals: {', '.join(people_embs.keys())}")
    
    process_event_photos(
        app, 
        people_embs, 
        photos_dir, 
        out_dir, 
        args.threshold, 
        args.min_face
    )

    print("\nDone!")
    print(f"Results saved in: {out_dir}")
    print(f"CSV Report: {out_dir / 'reports' / 'matches.csv'}")
    print(f"JSON Report: {out_dir / 'reports' / 'matches.json'}")


if __name__ == "__main__":
    main()
