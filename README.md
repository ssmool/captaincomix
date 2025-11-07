![Python CAPTAIN_COMIX](./assets/captain_comix.gif)

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

### EXAMPLES TO IMAGE COMPOSITION:

The following are two examples for a `README.md` file on GitHub, demonstrating how to use the `captomix` method from the `captaincomix` library to create `.mp4` files with REMbg-processed images and video backgrounds.

-----

## `captaincomix` Usage Examples

These examples assume you have the `captaincomix` library installed and a file named `astronaut.png` and `pumpkin_halloween.png` available. The library is expected to automatically perform the **REMbg** (background removal) operation on the input image before placing it on the video background.

### 1\. Astronaut on a Google Space Background

This example uses an astronaut image, searches for a video background using the **`google`** provider with a context of 'space picture', and creates up to two videos (`max_videos=2`).

```python
from typing import List, Optional, Dict
# Assuming the captomix function is correctly imported
from captain_cominx.captomix import captomix

# Note: The search term for the background video is typically inferred by the
# library or specified via a dedicated argument not present in the provided signature.
# We set 'google' as the search provider as requested.

def example_astronaut_space():
    """
    Creates an MP4 file with an astronaut image over a space background
    found via Google.
    """
    print("--- Running Example 1: Astronaut in Space ---")
    
    result_astronaut = captomix(
        input_image_path="astronaut.png",
        output_path_prefix="astronaut_in_space",
        search_providers=['google'], # Search for background video using Google
        max_videos=2,                 # Create up to 2 videos
        filters=None                  # No special video effect filters
    )
    
    print("Example 1 completed. Output details:")
    print(result_astronaut)

if __name__ == '__main__':
    example_astronaut_space()
```

-----

### 2\. Halloween Pumpkin with Snow and Storm Filters

This example demonstrates two sequential calls: one to add the **`snow`** filter and another to add the **`storm`** filter, both using a background video of 'Hallowen monsters' searched on **`flickr`**. The `max_videos` limit is set to **10**.

```python
from typing import List, Optional, Dict
from captain_cominx.captomix import captomix

def example_halloween_filters():
    """
    Creates two MP4 files from a pumpkin image over a Halloween monsters
    background (Flickr), with 'snow' and 'storm' effects applied.
    """
    input_img = "pumpkin_halloween.png"
    search_providers = ['flickr'] # Search for 'Hallowen monsters' background on Flickr
    max_vids = 10                  # Set max videos to 10
    
    # --- Execution 1: Add Snow Filter ---
    print("\n--- Running Example 2, Execution 1: Pumpkin with Snow Filter ---")
    
    result_pumpkin_snow = captomix(
        input_image_path=input_img,
        output_path_prefix="pumpkin_halloween_monsters_snow_final_movie.mp4",
        search_providers=search_providers,
        max_videos=max_vids,
        filters=['snow'] # Apply the snow effect filter
    )
    
    print("Execution 1 completed. Output details (Snow):")
    print(result_pumpkin_snow)

    # --- Execution 2: Add Storm Filter ---
    # This execution uses the same base image and background search but applies a different filter.
    print("\n--- Running Example 2, Execution 2: Pumpkin with Storm Filter ---")
    
    # The prompt specified "storm by the google filter". Assuming 'google' refers to the filter name source,
    # but practically, we use the 'storm' effect filter.
    result_pumpkin_storm = captomix(
        input_image_path=input_img,
        output_path_prefix="pumpkin_halloween_monsters_storm_final_movie.mp4",
        search_providers=search_providers,
        max_videos=max_vids,
        filters=['storm'] # Apply the storm effect filter
    )
    
    print("Execution 2 completed. Output details (Storm):")
    print(result_pumpkin_storm)

if __name__ == '__main__':
    example_halloween_filters()
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
