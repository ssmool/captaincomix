# CaptainComix

**CaptainComix** is a Python tool to transform still images into animated sequences by overlaying video/GIF layers and applying visual filters.  
Inspired by glitch-comics and collage FX aesthetics.

Created by **#asytric**  
Contact: **eusmool@gmail.com**
---

## ✨ Features

- Merge **images** with **GIF/MP4** motion layers.
- Control number of motion layers applied.
- Apply visual effects (filters).
- Export final output as MP4 or GIF.
- Script or command-line usage.

---

## 🛠 Installation

```bash
git clone https://github.com/ssmool/captaincomix
cd captaincomix
pip install -r requirements.txt
```

Here is a clean **`README.md`** you can copy directly into your repository:

````markdown
# CaptainComix

**CaptainComix** is a Python tool to transform still images into animated sequences by overlaying video/GIF layers and applying visual filters.  
Inspired by glitch-comics and collage FX aesthetics.

Created by **#asytric**  
Contact: **eusmool@gmail.com**

---

## ✨ Features

- Merge **images** with **GIF/MP4** motion layers.
- Control number of motion layers applied.
- Apply visual effects (filters).
- Export final output as MP4 or GIF.
- Script or command-line usage.

---

## 🛠 Installation

```bash
git clone https://github.com/ssmool/captaincomix
cd captaincomix
pip install -r requirements.txt
````

---

## 🚀 Command Line Usage

```bash
python captaincomix.py [-h] [--out OUT] [--max_videos MAX_VIDEOS] [--filters [FILTERS ...]] input_image
```

### **Arguments**

| Argument       | Description                         | Default           |
| -------------- | ----------------------------------- | ----------------- |
| `input_image`  | Source image to transform           | *required*        |
| `--out`, `-o`  | Output filename (without extension) | `captomix_output` |
| `--max_videos` | How many video/GIF layers to merge  | `2`               |
| `--filters`    | List of filters to apply            | `BRIGHT`          |

### **Example**

```bash
python captaincomix.py myimage.jpg --out final_movie --max_videos 3 --filters BLUR CONTRAST SHARPEN
```

This will generate:

```
final_movie.mp4
```

---

## 🧩 Available Filters

| Filter         | Description                     |
| -------------- | ------------------------------- |
| `BLUR`         | Softens the image               |
| `CONTOUR`      | Enhances edges                  |
| `DETAIL`       | Increases small visual detail   |
| `SHARPEN`      | Sharpens the overall image      |
| `BRIGHTNESS:x` | Adjust brightness (`x` = level) |
| `CONTRAST:x`   | Adjust contrast (`x` = level)   |
| `COLOR:x`      | Adjust saturation (`x` = level) |
| `GRAYSCALE`    | Convert to black & white        |

**Example:**

```bash
--filters BRIGHTNESS:1.3 CONTRAST:0.8 GRAYSCALE
```

---

## 🐍 Python Module Usage

You can import and use it in your own projects:

```python
from captain_cominx.captomix import captomix

src = captomix(
    image_file="myimage.png",      # input image
    save_out_mp4="output.mp4",     # output movie
    max_search=2,                  # number of GIF/MP4 layers
    filters_fx=["BLUR", "BRIGHTNESS:1.2", "SHARPEN"]
)

or

from captain_cominx.captomix import captomix

__r__mp4__ = captomix(image_file,save_out_mp4,max_search,filters_fx)

```

---

## 📜 License

MIT — free to use and remix.

---

## 🎨 Credits

**CaptainComix** by **#asytric**
Contact: **[eusmool@gmail.com](mailto:eusmool@gmail.com)**

```

---

If you'd like, I can now:

✅ Generate example screenshots  
✅ Create a demo GIF  
✅ Build a clearer filter preview table  
✅ Write a PyPI `setup.py` for packaging  

Just tell me!
```
