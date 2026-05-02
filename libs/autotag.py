import sys
import subprocess
from pathlib import Path
import argparse
import json
import os
from collections import defaultdict
import msvcrt

os.system("")

BASE_PATH = Path(__file__).resolve().parent.parent
LIBS_PATH = Path(__file__).resolve().parent

LIBS_PATH.mkdir(exist_ok=True)
sys.path.append(str(LIBS_PATH))


def ensure_package(package, import_name=None):
    try:
        __import__(import_name or package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(LIBS_PATH),
            package
        ])


ensure_package("pillow", "PIL")
ensure_package("numpy")
ensure_package("tqdm")
ensure_package("huggingface_hub")
ensure_package("onnxruntime")

from PIL import Image
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import csv
from tqdm import tqdm

HF_CACHE = BASE_PATH / "libs" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def load_config(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_blacklist(path):
    if not path.exists():
        return set()

    result = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            clean = line.lower()
            result.add(clean)
            result.add(clean.replace(" ", "_"))

    return result


def load_model(repo, model_file, label_file):
    try:
        model_path = hf_hub_download(repo_id=repo, filename=model_file, local_files_only=True)
        label_path = hf_hub_download(repo_id=repo, filename=label_file, local_files_only=True)
        print("Model loaded from cache")
    except Exception:
        print("Downloading model...")
        model_path = hf_hub_download(repo_id=repo, filename=model_file)
        label_path = hf_hub_download(repo_id=repo, filename=label_file)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    tags = []
    with open(label_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            tags.append(row[1])

    return session, tags


def preprocess(img):
    img = img.resize((448, 448))
    arr = np.array(img).astype(np.float32)
    arr = arr[:, :, ::-1]
    return np.expand_dims(arr, 0)


def process_image(file, resize_enabled, output_folder, target_size):
    img = Image.open(file).convert("RGB")

    if resize_enabled:
        w, h = img.size
        scale = target_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        output_image = output_folder / file.name
        img.save(output_image, quality=85, optimize=True)

        txt_path = output_image.with_suffix(".txt")

        src_txt = file.with_suffix(".txt")
        if src_txt.exists() and not txt_path.exists():
            txt_path.write_text(src_txt.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        txt_path = file.with_suffix(".txt")

    return img, txt_path


def predict_batch(session, tags, images, threshold):
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: images})[0]

    return [
        [tag for tag, prob in zip(tags, pred) if prob > threshold]
        for pred in preds
    ]


def run_batch(session, tags, images, txt_paths, threshold, trigger, blacklist):
    images_np = np.vstack(images)
    results = predict_batch(session, tags, images_np, threshold)

    for txt_path, result in zip(txt_paths, results):
        filtered = []

        for t in result:
            t_lower = t.lower()
            t_spaced = t_lower.replace("_", " ")

            if t_lower in blacklist or t_spaced in blacklist:
                continue
            if any(b in t_spaced for b in blacklist):
                continue
            if any(b in t_spaced.split() for b in blacklist):
                continue

            filtered.append(t_spaced)

        if txt_path.exists():
            existing = [x.strip() for x in txt_path.read_text(encoding="utf-8").split(",") if x.strip()]
            if trigger and trigger not in existing:
                existing.insert(0, trigger)
            final_tags = existing
        else:
            final_tags = [trigger] + filtered if trigger else filtered

        txt_path.write_text(", ".join(final_tags), encoding="utf-8")


def rename_duplicates(files):
    groups = defaultdict(list)
    for f in files:
        groups[f.stem].append(f)

    duplicates = {k: v for k, v in groups.items() if len(v) > 1}

    if not duplicates:
        return False

    YELLOW = "\033[33m"
    RESET = "\033[0m"

    print("\nDuplicate filenames detected:")
    for k, v in duplicates.items():
        print(f"  {k}: {[x.name for x in v]}")

    print(f"\n{YELLOW}WARNING: image files must have a unique name.{RESET}")
    print("Press Enter to rename duplicates or press Esc to exit.")

    while True:
        key = msvcrt.getch()
        if key == b'\r':
            break
        elif key == b'\x1b':
            print("Exiting.\n")
            sys.exit()

    print("\nRenaming duplicate filenames...")

    for stem, files in duplicates.items():
        files_sorted = sorted(files, key=lambda x: x.name)

        for i, f in enumerate(files_sorted, start=1):
            new_name = f"{stem}_{i}{f.suffix}"
            new_path = f.with_name(new_name)
            f.rename(new_path)

    print("Renaming done.\n")
    return True


def process_folder(folder, args, config, session, tags):
    folder = Path(folder).expanduser().resolve()

    if not folder.exists():
        print("Folder not found")
        return

    threshold = args.threshold or config.get("threshold", 0.35)

    resize_enabled = config.get("resize", True)
    user_input = input("Enable resize? (y/n): ").strip().lower()
    if user_input == "y":
        resize_enabled = True
    elif user_input == "n":
        resize_enabled = False

    batch_size = config.get("batch_size", 16)
    target_size = config.get("target_size", 1024)

    output_folder_name = config.get("output_folder_name", "dataset")
    extensions = config.get("extensions", [".jpg", ".jpeg", ".png", ".webp"])

    blacklist = load_blacklist(BASE_PATH / "blacklist" / "blacklist.txt")

    output_folder = folder / output_folder_name if resize_enabled else None
    if output_folder:
        output_folder.mkdir(exist_ok=True)

    while True:
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
        if not rename_duplicates(image_files):
            break

    batch_images = []
    batch_txt = []

    for file in tqdm(image_files):
        try:
            img, txt_path = process_image(file, resize_enabled, output_folder, target_size)
            batch_images.append(preprocess(img))
            batch_txt.append(txt_path)

            if len(batch_images) == batch_size:
                run_batch(session, tags, batch_images, batch_txt, threshold, args.trigger, blacklist)
                batch_images.clear()
                batch_txt.clear()

        except Exception as e:
            print(f"Error with {file.name}: {e}")

    if batch_images:
        run_batch(session, tags, batch_images, batch_txt, threshold, args.trigger, blacklist)

    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--trigger")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--config")

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else BASE_PATH / "config" / "config.json"
    config = load_config(config_path)

    session, tags = load_model(
        config.get("model_repo"),
        config.get("model_file"),
        config.get("label_file")
    )

    current_folder = args.folder

    while True:
        process_folder(current_folder, args, config, session, tags)

        print("Press Enter to continue tagging or press Esc to exit...")

        while True:
            key = msvcrt.getch()
            if key == b'\r':
                break
            elif key == b'\x1b':
                return

        next_folder = input("Enter next folder path: ").strip()
        if not next_folder:
            break

        new_trigger = input("Enter trigger tag (or leave empty): ").strip()
        args.trigger = new_trigger if new_trigger else None

        current_folder = next_folder


if __name__ == "__main__":
    main()