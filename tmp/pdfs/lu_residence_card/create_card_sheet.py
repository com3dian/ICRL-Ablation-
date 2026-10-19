from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


ROOT = Path("/home/com3dian/Github/ICRL-Ablation-")
WORK = ROOT / "tmp" / "pdfs" / "lu_residence_card"
OUTPUT = ROOT / "output" / "pdf" / "鲁泽昊_居留卡_正反面单页竖向叠放.pdf"
IMAGES = [WORK / "card-000.jpg", WORK / "card-001.jpg"]


def rotated_size(path: Path) -> tuple[int, int]:
    """Return the pixel size after rotating the portrait scan 90 degrees CCW."""
    with Image.open(path) as image:
        return image.height, image.width


def draw_rotated_ccw(
    pdf: canvas.Canvas,
    image_path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Draw a JPEG rotated 90 degrees CCW without re-encoding it."""
    pdf.saveState()
    pdf.translate(x, y)
    pdf.rotate(90)
    pdf.drawImage(
        str(image_path),
        0,
        -width,
        width=height,
        height=width,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.restoreState()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    page_width, page_height = A4
    margin = 10 * mm
    gap = 5 * mm
    max_width = page_width - 2 * margin
    slot_height = (page_height - 2 * margin - gap) / 2

    pdf = canvas.Canvas(str(OUTPUT), pagesize=(page_width, page_height))
    pdf.setTitle("鲁泽昊居留卡正反面 - 单页竖向叠放")

    image_sizes = [rotated_size(image_path) for image_path in IMAGES]
    narrowest_aspect = min(width / height for width, height in image_sizes)
    common_width = min(max_width, slot_height * narrowest_aspect)

    for index, (image_path, (pixel_width, pixel_height)) in enumerate(
        zip(IMAGES, image_sizes)
    ):
        draw_width = common_width
        draw_height = common_width * pixel_height / pixel_width

        x = (page_width - draw_width) / 2
        slot_y = margin + (1 - index) * (slot_height + gap)
        y = slot_y + (slot_height - draw_height) / 2
        draw_rotated_ccw(pdf, image_path, x, y, draw_width, draw_height)

    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
