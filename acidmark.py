from matplotlib import font_manager
from PIL import Image, ImageFont, ImageDraw
import argparse
import Augmentor
import glob
import tempfile
import math
import numpy as np
import os


def grow_box(text, font_file, container_size):
    """Finds the max the font and bounding box size that still fits in the container."""

    font_size = 0

    while True:
        font_size += 1
        font = ImageFont.truetype(font_file, font_size)

        box_size = (
            max(font.getsize(t)[0] for t in text),
            sum(font.getsize(t)[1] for t in text),
        )

        if any(i > j for i, j in zip(box_size, container_size)):
            break

    return font_size, box_size


def generate_text(
    text, container_size, alpha=100, rotate=False, font_family=None, box_color=None
):
    """Creates an image with text."""

    diagonal = int(math.sqrt(container_size[0] ** 2 + container_size[1] ** 2))

    font_file = font_manager.findfont(
        font_manager.FontProperties(family=font_family, weight="bold")
    )
    if font_family:
        print(f"Selecting font {font_file}")

    # the box size may run off the edges after rotation
    font_size, box_size = grow_box(text, font_file, container_size)
    font = ImageFont.truetype(font_file, font_size)

    img = Image.new("RGBA", box_size, box_color if box_color else (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    pos = [0, 0]
    for t in text:
        pos[0] = (box_size[0] - font.getsize(t)[0]) // 2
        draw.text(pos, t, fill=(0, 0, 0, alpha), font=font)
        pos[1] += font.getsize(t)[1]

    if rotate:
        angle = 90 - math.degrees(math.acos(container_size[1] / diagonal))
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    return img


def add_noise(img, mean=100, var=10):
    """Varies the alpha channel of pixels with a transparency close to mean."""

    alpha = np.array(img.getchannel("A"))

    # generate an array of normally distributed discrete values
    random = np.random.normal(mean, var, alpha.shape).clip(0, 255).astype("uint8")

    # replace pixels with an alpha level close to the mean to keep smooth edges
    np.putmask(alpha, max(mean - 10, 0) < alpha, random)

    img.putalpha(Image.fromarray(alpha))
    return img


def distort(watermark, out_image, dist):
    """Generate random image perturbations using Augmentor."""

    # Augmentor does not support loading images directly yet, so we save/load from disk.
    with tempfile.TemporaryDirectory() as tmp_dir:
        watermark.save(os.path.join(tmp_dir, "text.png"), "png")

        augmented_dir = os.path.join(tmp_dir, "augmented")

        p = Augmentor.Pipeline(tmp_dir, augmented_dir, "png")
        p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=dist)
        p.sample(1)

        watermark = Image.open((glob.glob(os.path.join(augmented_dir, "*.png")))[0])

    return watermark


def paste(canvas, watermark):
    """Pastes the watermark in the middle of the canvas."""

    offset = tuple((x - y) // 2 for x, y in zip(canvas.size, watermark.size))
    canvas.paste(watermark, offset, watermark)
    return canvas


def main(args):
    if args.output:
        out_image = args.output
    else:
        out_image = "{}_marked.jpg".format(os.path.splitext(args.input)[0])

    base = Image.open(args.input)
    watermark = generate_text(
        args.text.splitlines(),
        base.size,
        args.opacity,
        True,
        args.font,
        None,
    )

    if args.distort:
        watermark = distort(watermark, out_image, args.distort)

    if args.noise:
        watermark = add_noise(watermark, args.opacity, args.noise)

    output = paste(base, watermark)
    output.save(out_image, "jpeg", quality=args.jpeg_quality)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input", type=str, help="input image")
    parser.add_argument(
        "text", type=str, help="watermark text, may include newline characters"
    )

    parser.add_argument(
        "-d", "--distort", type=int, default=10, help="distortion factor"
    )
    parser.add_argument("--font", type=str, help="font family to use")
    parser.add_argument(
        "--jpeg-quality", type=int, default=75, help="JPEG output quality [0-100]"
    )
    parser.add_argument(
        "-n",
        "--noise",
        type=int,
        default=10,
        help="variance in the opacity of watermark pixels",
    )
    parser.add_argument(
        "--opacity", type=int, default=100, help="opacity of watermark [0-255]"
    )
    parser.add_argument("-o", "--output", type=str, help="output image, always JPEG")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
