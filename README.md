# CoinRecog - Task 1 Preprocessing

This repository currently implements the **first stage** of the Smart Currency Recognition pipeline (before contour detection):

1. Load image
2. Resize image
3. Convert to grayscale
4. Apply Gaussian blur
5. Apply Otsu threshold
6. Return processed mask

## Run

```bash
python3 preprocess_currency.py <path_to_image> --width 800 --out-dir outputs
```

### Outputs

- `outputs/resized_image.png` → original resized image
- `outputs/clean_binary_mask.png` → clean binary mask

## Dependency

```bash
pip install opencv-python numpy
```
