import numpy as np
import io
import base64
from PIL import Image

def convert_array_to_data_url_png(array: np.ndarray) -> str:
    """Convert a numpy array to a base64-encoded PNG data URL."""
    converted_array = array
    if converted_array.dtype != np.uint8:
        converted_array = np.clip(converted_array, 0, 255).astype(np.uint8)

    mode = "L" if converted_array.ndim == 2 else "RGB"
    image = Image.fromarray(converted_array, mode=mode)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/png;base64,{base64_encoded}"

def convert_rgb_to_grayscale_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using standard luminance weights."""
    red_channel = image_rgb[..., 0].astype(np.float32)
    green_channel = image_rgb[..., 1].astype(np.float32)
    blue_channel = image_rgb[..., 2].astype(np.float32)

    # Standard RGB to grayscale conversion weights (ITU-R BT.601)
    grayscale = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
    return np.clip(grayscale, 0, 255).astype(np.uint8)

