import sys
import subprocess
from pathlib import Path
import os

os.system("")

BASE_PATH = Path(__file__).resolve().parent
LIBS_PATH = BASE_PATH / "libs"

LIBS_PATH.mkdir(exist_ok=True)
sys.path.append(str(LIBS_PATH))


def ensure_package(package, import_name=None):
    try:
        __import__(import_name or package)
    except ImportError:
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
ensure_package("imagehash")

from PIL import Image
import imagehash


def make_link(file):
    uri = file.resolve().as_uri()
    return f"\033]8;;{uri}\033\\{file.name}\033]8;;\033\\"


def main():
    folder_input = input("Enter folder path: ").strip().strip('"')
    folder = Path(folder_input)

    if not folder.exists():
        return

    extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    hashes = {}
    threshold = 8
    found = False

    for file in image_files:
        try:
            img = Image.open(file).convert("RGB")
            h = imagehash.phash(img)

            matched = False

            for existing_hash, group in hashes.items():
                if h - existing_hash <= threshold:
                    group.append(file)
                    matched = True
                    break

            if not matched:
                hashes[h] = [file]

        except:
            pass

    group_id = 1

    for group in hashes.values():
        if len(group) > 1:
            found = True

            print(f"\n[Group {group_id}]\n")

            for file in group:
                print(make_link(file))

            group_id += 1

    if found:
        yellow = "\033[33m"
        reset = "\033[0m"

        print(f"\n{yellow}Possible duplicates found. Please review manually.")
        print("Ctrl+Left click file names to open images." + reset + "\n")


if __name__ == "__main__":
    main()