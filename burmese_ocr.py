import pytesseract
from PIL import Image
from os import path

base = r"C:\Users\User\Pictures\Moemyintaung\m"


def as_pdf(image, out):
    # Get a searchable PDF
    pdf = pytesseract.image_to_pdf_or_hocr(image, extension="pdf")
    with open(f"{out}.pdf", "w+b") as f:
        f.write(pdf)  # pdf type is bytes by default


def main(args):
    image = base
    # join base path with args.image
    image = path.join(image, args.image)

    text = pytesseract.image_to_string(image, lang=args.lang)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)


def parse_arg():
    import argparse

    parser = argparse.ArgumentParser(description="Burmese OCR")
    parser.add_argument(
        "-i", "--image", type=str, help="Image file name", required=True
    )
    parser.add_argument(
        "-o", "--out", type=str, help="Output file path", default="out.txt"
    )
    parser.add_argument(
        "-l", "--lang", type=str, help="Parsing Language", default="mya+eng"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(parse_arg())
