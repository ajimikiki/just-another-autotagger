![Screenshot](https://i.imgur.com/xoKlfJW.png)
# Just Another Autotagger

Minimal CLI tool for tagging and resizing image datasets.

## Features

- Tag images using the wd-swinv2-tagger-v3 model
- Prepend existing .txt files with a trigger word
- Resize images (1024px by default; resized images are saved in the dataset folder)
- Blacklist certain tags from being added to .txt (edit blacklist.txt)
- Rename images that have the same name but different extensions

## Usage

Run setup.bat once to install everything. All dependencies will be installed in the libs folder and will use about 600 MB of disk space.

After that, use run.bat.

## Changelog

Version 1.1.0
- Added cull and cull-whitelist functionality
- Cull removes less specific tags (e.g., "dress" → removed when "china dress" exists)
- Skip tagging when .txt exists (resize, trigger, and cull still apply)

## Credits

This project uses the wd-swinv2-tagger-v3 model by SmilingWolf:  
https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3
