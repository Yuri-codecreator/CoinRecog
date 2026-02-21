"""Currency image preprocessing utilities.

Implements the first stage of the project pipeline:
- load image
- resize image
- convert to grayscale
- apply gaussian blur
- apply Otsu threshold
- return processed mask
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def preprocess_currency_image(
    image_path: str | Path,
    target_width: int = 800,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess a currency image.

    Args:
        image_path: Path to image file.
        target_width: Width in pixels for resized output. Aspect ratio is preserved.

    Returns:
        A tuple of:
        - resized_color: Original resized BGR image.
        - clean_binary_mask: Binary mask after blur + Otsu threshold.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If image cannot be decoded or width is invalid.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if target_width <= 0:
        raise ValueError("target_width must be > 0")

    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not decode image file: {image_path}")

    h, w = original.shape[:2]
    scale = target_width / float(w)
    target_height = int(h * scale)

    resized_color = cv2.resize(
        original,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, clean_binary_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return resized_color, clean_binary_mask


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess Philippine currency image and output resized image + binary mask"
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--width", type=int, default=800, help="Target width for resized output"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write output files",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    resized, mask = preprocess_currency_image(args.image, target_width=args.width)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    resized_path = args.out_dir / "resized_image.png"
    mask_path = args.out_dir / "clean_binary_mask.png"

    cv2.imwrite(str(resized_path), resized)
    cv2.imwrite(str(mask_path), mask)

    print(f"Saved resized image: {resized_path}")
    print(f"Saved clean binary mask: {mask_path}")


if __name__ == "__main__":
    main()
