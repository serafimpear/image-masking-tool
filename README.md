
# Image Masking Tool

A Python/OpenCV-based application for quickly creating binary masks by drawing directly on images.

## Features

* Draw mask areas with left-click drag
* Erase mask regions with right-click drag
* Navigate between images with arrow keys
* Real-time preview of drawing
* Auto-save masks as `<image_filename>.mask.png`
* Preserves aspect ratio and supports window resizing
* Top menu with: All White, All Black, Reset, Undo, Redo, Remove Mask

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/serafimpear/image-masking-tool.git
   cd image-masking-tool
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the mask creator against a folder of images:

```bash
python mask_creator.py <image_folder>
```

Replace `<image_folder>` with the path containing your PNG/JPG images.

## Generative AI Notice

This tool and its documentation were authored with the assistance of a generative AI to accelerate development and ensure clarity.

## Example Use Case

This masking tool can be especially helpful when working with RealityCapture, which does not offer built-in masking functionality. You can prepare masks here and then import them into RealityCapture to guide processing.

## License

Licensed under the MIT License. See [LICENSE](https://chatgpt.com/c/LICENSE) for details.
