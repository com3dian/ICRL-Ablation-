from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


SOURCE_DIR = Path(
    "/home/com3dian/Downloads/鲁泽昊_张逸棹_结婚证件/结婚证件"
)
ROOT = Path("/home/com3dian/Github/ICRL-Ablation-")
OUTPUT_DIR = ROOT / "output" / "pdf" / "证件_实际尺寸"
EXTRACT_ROOT = ROOT / "tmp" / "pdfs" / "card_actual_size" / "extracted"

SOURCE_NAMES = [
    "张逸棹_居留卡.pdf",
    "张逸棹_居留卡 (Copy).pdf",
    "张逸棹_身份证.pdf",
    "鲁泽昊_居留卡.pdf",
    "鲁泽昊_居留卡 (Copy).pdf",
    "鲁泽昊_身份证.pdf",
]

# ISO/IEC 7810 ID-1: the physical size used by Chinese identity cards and
# Dutch residence-permit cards.
CARD_WIDTH = 85.60 * mm
CARD_HEIGHT = 53.98 * mm
VERTICAL_GAP = 18 * mm
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".ppm", ".pbm"}


def extract_images(source: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    prefix = destination / "side"
    subprocess.run(
        ["pdfimages", "-all", str(source), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    images = sorted(
        path for path in destination.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) != 2:
        raise RuntimeError(f"Expected exactly two card-side images in {source}, got {len(images)}")
    return images


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def draw_rotated_ccw(
    pdf: canvas.Canvas,
    image_path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    pdf.saveState()
    pdf.translate(x, y)
    pdf.rotate(90)
    pdf.drawImage(
        str(image_path),
        0,
        -width,
        width=height,
        height=width,
        preserveAspectRatio=False,
        mask="auto",
    )
    pdf.restoreState()


def draw_card_side(
    pdf: canvas.Canvas,
    image_path: Path,
    box_y: float,
    page_width: float,
) -> None:
    pixel_width, pixel_height = image_dimensions(image_path)
    rotate = pixel_height > pixel_width
    draw_width = CARD_WIDTH
    draw_height = CARD_HEIGHT
    x = (page_width - draw_width) / 2
    y = box_y

    if rotate:
        draw_rotated_ccw(pdf, image_path, x, y, draw_width, draw_height)
    else:
        pdf.drawImage(
            str(image_path),
            x,
            y,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=False,
            mask="auto",
        )


def create_output(source: Path, images: list[Path], output: Path) -> None:
    page_width, page_height = A4
    block_height = 2 * CARD_HEIGHT + VERTICAL_GAP
    bottom_y = (page_height - block_height) / 2
    top_y = bottom_y + CARD_HEIGHT + VERTICAL_GAP

    pdf = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    pdf.setTitle(f"{source.stem} - 实际证件尺寸")
    draw_card_side(pdf, images[0], top_y, page_width)
    draw_card_side(pdf, images[1], bottom_y, page_width)
    pdf.showPage()
    pdf.save()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    EXTRACT_ROOT.mkdir(parents=True)

    for index, source_name in enumerate(SOURCE_NAMES):
        source = SOURCE_DIR / source_name
        extract_dir = EXTRACT_ROOT / f"{index:02d}"
        images = extract_images(source, extract_dir)
        output = OUTPUT_DIR / f"{source.stem}_实际尺寸_单页.pdf"
        create_output(source, images, output)
        print(output)


if __name__ == "__main__":
    main()
