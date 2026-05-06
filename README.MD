![Screenshot](https://i.imgur.com/2spbvIY.png)
# Just Another Autotagger

Minimal CLI tool for tagging and resizing image datasets.
## Features

- Tag images using the `wd-swinv2-tagger-v3` model
- Prepend existing `.txt` captions with a trigger word
- Resize images (1024px by default; output is saved to the `dataset` folder)
- Automatically convert obscure image formats (`.avif`, `jfif`, `.heic`) to `.jpg`
- Exclude unwanted tags using `blacklist.txt`
- Automatically rename files with identical base names but different extensions
- Cull redundant or overlapping tags
- Detect duplicate or near-duplicate images

## Usage

Run `setup.bat` once to install all dependencies (~600 MB, installed into the `libs` folder).

After that, run `run.bat`.

## Changelog

### Version 1.2.0

- Images in formats unsupported by many training workflows (`.avif`, `.heic`) are now automatically converted to `.jpg` before tagging. Original files are moved to the `unsupported_originals` folder.
- Added duplicate image detection to help identify visually identical or near-identical images in datasets
- Duplicate detection can be disabled in `config.json`. Change `"dupe_search": true` to `"dupe_search": false`

### Version 1.1.0

- Added `cull` and `cull_whitelist` functionality  
- To enable cull, set `"auto_cull": true` in `config.json`  
- Cull removes less specific tags when a more specific tag is present  
  (e.g., `dress` is removed if `china dress` exists)
- Added option to skip tagging if a `.txt` file already exists  
  (resize, trigger, and cull still apply)

This project uses the wd-swinv2-tagger-v3 model by SmilingWolf:  
https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3
