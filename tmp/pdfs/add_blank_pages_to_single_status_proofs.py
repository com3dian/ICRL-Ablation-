from pathlib import Path

from pypdf import PdfReader, PdfWriter


SOURCE_DIR = Path(
    "/home/com3dian/Downloads/鲁泽昊_张逸棹_结婚证件/结婚证件"
)
OUTPUT_DIR = Path(
    "/home/com3dian/Github/ICRL-Ablation-/output/pdf/单身证明_单面打印"
)
SOURCE_NAMES = ["张逸棹_单身证明.pdf", "鲁泽昊_单身证明.pdf"]


def add_blank_after_each_page(source: Path, output: Path) -> None:
    reader = PdfReader(source)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)
        writer.add_blank_page(
            width=float(page.mediabox.width),
            height=float(page.mediabox.height),
        )

    writer.add_metadata(
        {
            "/Title": f"{source.stem} - 内容页后加空白页",
            "/Subject": "用于双面打印设置下实现内容单面输出",
        }
    )
    with output.open("wb") as stream:
        writer.write(stream)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source_name in SOURCE_NAMES:
        source = SOURCE_DIR / source_name
        output = OUTPUT_DIR / f"{source.stem}_内容页后加空白页.pdf"
        add_blank_after_each_page(source, output)
        print(output)


if __name__ == "__main__":
    main()
