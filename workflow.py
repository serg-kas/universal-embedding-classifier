"""
Функции рабочего процесса
"""

import numpy as np
import cv2 as cv
# from PIL import Image, ImageDraw, ImageFont
# from pdf2image import convert_from_path, convert_from_bytes  # sudo apt-get install poppler-utils (рендеринг страниц)
# from PyPDF2 import PdfReader, PdfWriter                      # pip install PyPDF2 (разбиение pdf на страницы)
# from fpdf import FPDF                                        # pip install fpdf2 (создание pdf)
# import fitz                                                  # pip install PyMuPDF (разбиение на страницы, рендеринг)
#
# import io
# import os
# import sys
import time
# from multiprocessing import Process, Queue
#
# import json
# import base64
# import requests
# import magic
# import pika
# from qreader import QReader
#
# import pytesseract                                           # sudo apt install tesseract-ocr
# import Levenshtein
# import re
# from pprint import pprint, pformat
#
# from classify_cnnfe import Classifier
import settings as s
# import helpers.dnn as dnn
import helpers.utils as u

# Цвета
black = s.black
blue = s.blue
green = s.green
red = s.red
yellow = s.yellow
purple = s.purple
turquoise = s.turquoise
white = s.white


def get_images_simple(source_files, bgr_to_rgb=None, verbose=False):
    """
    По списку файлов считывает изображения и возвращает их списком.
    Упрощенный вариант без авторотации

    :param source_files: список файлов с полными путями к ним
    :param bgr_to_rgb: преобразовывать цветное изображение в RGB
    :param verbose: выводить подробные сообщения
    :return: список изображений
    """
    time_0 = time.perf_counter()

    result = []
    if True:
        if verbose:
            print("Загружаем изображения без авторотации")
        for f in source_files:
            img = cv.imread(f)
            if img is None:
                if verbose:
                    print("  Ошибка чтения файла: {} ".format(f))
                continue
            if bgr_to_rgb:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            result.append(img)

    time_1 = time.perf_counter()
    if verbose:
        print("Загружено изображений: {}, время {:.2f} с.".format(len(result), time_1 - time_0))
    return result

