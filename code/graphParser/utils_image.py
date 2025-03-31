import io
import base64
from PIL import Image
import pymupdf
import os
from typing import Tuple

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()
        file_ext = os.path.splitext(image_path)[1].lower()

        if file_ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif file_ext == '.png':
            mime_type = 'image/png'
        else:
            mime_type = 'image/unknown'

        return base64.b64encode(image_content).decode("utf-8"), mime_type


def encode_pil_image(image, format: str = 'PNG') -> Tuple[str, str]:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    mime_type = f"image/{format.lower()}"
    return image_base64, mime_type


def crop_image(img, coordinates, output_path):
    """
    이미지를 주어진 좌표에 따라 자르고 저장하는 정적 메서드

    :param img: 원본 이미지 객체
    :param coordinates: 정규화된 좌표 (x1, y1, x2, y2)
    :param output_path: 저장할 파일 경로
    """
    img_width, img_height = img.size
    
    x1, y1, x2, y2 = [
        int(coord * dim)
        for coord, dim in zip(coordinates, [img_width, img_height] * 2)
    ]
    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img.save(output_path)


def pdf_to_image(pdf_file, page_num, dpi=300):
    """
    PDF 파일의 특정 페이지를 이미지로 변환하는 메서드

    :param page_num: 변환할 페이지 번호 (1부터 시작)
    :param dpi: 이미지 해상도 (기본값: 300)
    :return: 변환된 이미지 객체
    """

    with pymupdf.open(pdf_file) as doc:
        page = doc[page_num].get_pixmap(dpi=dpi)
        target_page_size = [page.width, page.height]
        page_img = Image.frombytes("RGB", target_page_size, page.samples)
    return page_img


def merge_images(image_paths):    
    images = [Image.open(img_path) for img_path in image_paths]
    width = min(img.width for img in images)
    images = [img.resize((width, img.height)) for img in images]
    merged = Image.new('RGB', (width, sum(img.height for img in images)))
    y_offset = 0
    for img in images:
        merged.paste(img, (0, y_offset))
        y_offset += img.height
    
    return merged