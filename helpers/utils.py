"""
Функции различного назначения
"""
import numpy as np
import math
import re
#
import cv2 as cv
from PIL import Image  # ImageDraw, ImageFont
from imutils import perspective, auto_canny
import matplotlib.pyplot as plt
#
import io
import os
# import sys
import inspect
#
import settings as s


# #############################################################
#                       ФУНКЦИИ OpenCV
# #############################################################
def autocontrast_cv(img, clip_limit=2.0, called_from=False):
    """
    Функция автокоррекции контраста методом
    CLAHE Histogram Equalization через цветовую модель LAB

    :param img: изображение
    :param clip_limit: порог используется для ограничения контрастности
    :param called_from: сообщать откуда вызвана функция
    :return: обработанное изображение
    """
    #
    if called_from:
        # текущий фрейм объект
        current_frame = inspect.currentframe()
        # фрейм объект, который его вызвал
        caller_frame = current_frame.f_back
        # у вызвавшего фрейма исполняемый в нём объект типа "код"
        code_obj = caller_frame.f_code
        # и получи его имя
        code_obj_name = code_obj.co_name
        print("Функция autocontrast_cv вызвана из:", code_obj_name)

    # converting to LAB color space
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color space
    result = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return result


def sharpen_image_cv(img, force_threshold=False):
    """
    Увеличивает резкость изображения
    с переходом в ч/б изображение и тресхолдом опционально.
    Использует фильтр (ядро) для улучшения четкости

    :param img: изображение
    :param force_threshold: тресхолд выходного изображение
    :return: обработанное изображение
    """
    img = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    img = cv.filter2D(img, -1, sharpen_kernel)

    if force_threshold:
        img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # show_image_cv(img, title="img sharpened")
    return img


def img_resize_cv(image, img_size=1024):
    """
    Функция ресайза картинки через opencv к заданному размеру по наибольшей оси

    :param image: исходное изображение
    :param img_size: размер, к которому приводить изображение
    :return: resized image
    """
    curr_h = image.shape[0]
    curr_w = image.shape[1]
    # Рассчитаем коэффициент для изменения размера
    if curr_w > curr_h:
        scale_img = img_size / curr_w
    else:
        scale_img = img_size / curr_h
    # Новые размеры изображения
    new_width = int(curr_w * scale_img)
    new_height = int(curr_h * scale_img)
    # делаем ресайз к целевым размерам
    image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    return image


def image_rotate_cv(image, angle, simple_way=False, resize_to_original=False):
    """
    Функция вращения картинки средствами opencv

    :param image: изображение
    :param angle: угол в градусах
    :param simple_way: упрощенный вариант с обрезкой краёв
    :param resize_to_original: ресайз результата к размерам исходного изображения
    :return:
    """
    height, width = image.shape[:2]

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width / 2, height / 2)

    # rotationMatrix = cv.getRotationMatrix2D((centreY, centreX), angle, 1.0)
    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

    if simple_way:
        # rotate image
        result = cv.warpAffine(image, rotation_mat, (width, height))
        return result

    else:
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to orig) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        result = cv.warpAffine(image, rotation_mat, (bound_w, bound_h))

        #
        if resize_to_original:
            result = cv.resize(result, (width, height), interpolation=cv.INTER_AREA)
            return result
        else:
            return result


def show_image_cv(img, title='Image '):
    """
    Выводит картинку на экран методом из opencv.
    Дожидается нажатия клавиши для продолжения

    :param img: изображение
    :param title: заголовок окна
    :return: none
    """
    if title == 'Image ':
        cv.imshow('Image ' + str(img.shape), img)
    else:
        cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_optimal_font_scale_cv(text, width):
    """
    Подбор оптимального размера шрифта для метода cv.putText

    :param text: текст для подбора шрифта
    :param width: нужная ширина
    :return: пододранные fontScale
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv.getTextSize(text, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=scale/10, thickness=3)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale/10
    return 1


def fig2img_cv(fig):
    """
    Convert a Matplotlib figure to a CV Image and return it
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=600)
    buf.seek(0)
    array = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    img = cv.imdecode(array, cv.IMREAD_COLOR)
    # img = Image.open(buf)  # если нужно изображение PIL
    return img


def get_img_encoded_list(img_list, extension=".png"):
    """
    Создание списка изображений png в памяти

    :param img_list: список изображений
    :param extension: тип изображения по расширению
    :return: png_list
    """
    encoded_list = []
    for img in img_list:
        img_encoded = cv.imencode(extension, img)[1]
        # encoded_list.append(img_encoded)
        encoded_list.append(img_encoded.tobytes())
    return encoded_list


def get_blank_img_cv(height, width, rgb_color=(0, 0, 0), txt_to_put=None):
    """
    Возвращает изображение заданной размерности, цвета, с заданной надписью

    :param height: высота
    :param width: ширина
    :param rgb_color: цвет RGB
    :param txt_to_put: текст
    :return: изображение
    """
    blank_img = np.zeros((height, width, 3), dtype=np.uint8)
    bgr_color = tuple(reversed(rgb_color))
    blank_img[:] = bgr_color
    #
    if txt_to_put is not None:
        X = int(width * 0.1)
        Y = int(height * 0.1)
        font_size = get_optimal_font_scale_cv(txt_to_put, int(width * 0.9))
        cv.putText(blank_img, txt_to_put, (X, Y), cv.FONT_HERSHEY_SIMPLEX, font_size, s.black, 2, cv.LINE_AA)
    return blank_img


def find_horizontal_cv(image,
                       region_coords,
                       blur_kernel_shape=(3, 3),
                       horizontal_kernel_shape=(40, 1),
                       horizontal_iterations=2,
                       min_y_step=30,
                       reverse_sort=False):
    """
    Функция поиска горизонтальных линий

    :param image: изображение
    :param region_coords: координаты региона для поиска горизонталей
    :param blur_kernel_shape: форма ядра для функции блюра
    :param horizontal_kernel_shape: форма ядра
    :param horizontal_iterations: итераций для поиска горизонталей
    :param min_y_step: минимальный шаг между линиями
    :param reverse_sort: обратная сортировка списка координат (нужна если ищем горизонтали снизу вверх)
    :return: список координат Y горизонтальных линий
    """
    # Регион где будем искать горизонтальные линии
    Rx1, Ry1, Rx2, Ry2 = region_coords
    region = image[Ry1:Ry2, Rx1:Rx2]
    #
    if 0 in region.shape:
        print("Некорректное изображение формы {}, горизонтали не могут быть найдены".format(region.shape))
        return []

    # Ищем горизонтальные линии
    gray = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, blur_kernel_shape, 0)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    #
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, horizontal_kernel_shape)
    detect_horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=horizontal_iterations)
    cnts = cv.findContours(detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #
    y_list = []  # список координат Y горизонтальных линий
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        # cv.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv.drawContours(region, [c], -1, (36, 255, 12), 2)

        if len(y_list) == 0:
            y_list.append(y + Ry1)
            # cv.circle(image, (100, y + Ry1), 10, (255, 0, 0), -1)

        if (len(y_list) > 0) and (abs(y_list[-1] - (y + Ry1)) >= min_y_step):  # линии не могут быть слишком близко
            y_list.append(y + Ry1)
            # cv.circle(image, (100, y + Ry1), 10, (255, 0, 0), -1)
    if reverse_sort:
        y_list.sort(reverse=True)
    else:
        y_list.sort()
    # print("y_list", y_list)
    return y_list


def find_vertical_cv(image,
                     region_coords,
                     blur_kernel_shape=(3, 3),
                     vertical_kernel_shape=(1, 40),
                     vertical_iterations=2,
                     min_x_step=30,
                     reverse_sort=False):
    """
    Функция поиска вертикальных линий

    :param image: изображение
    :param region_coords: координаты региона для поиска вертикалей
    :param blur_kernel_shape: форма ядра для функции блюра
    :param vertical_kernel_shape: форма ядра для морфологии
    :param vertical_iterations: итераций
    :param min_x_step: минимальный шаг между линиями
    :param reverse_sort: обратная сортировка списка координат (нужна если ищем вертикали справа налево)
    :return: список координат X вертикальных линий
    """
    # Регион где будем искать вертикальные линии
    Rx1, Ry1, Rx2, Ry2 = region_coords
    region = image[Ry1:Ry2, Rx1:Rx2]
    #
    if 0 in region.shape:
        print("Некорректное изображение формы {}, вертикали не могут быть найдены".format(region.shape))
        return []

    # Ищем вертикальные линии
    gray = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, blur_kernel_shape, 0)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    #
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, vertical_kernel_shape)
    detect_vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=vertical_iterations)
    cnts = cv.findContours(detect_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #
    x_list = []  # список координат X вертикальных линий
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        # cv.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv.drawContours(region, [c], -1, (36, 255, 12), 2)

        if len(x_list) == 0:
            x_list.append(x + Rx1)
            # cv.circle(image, (100, y + Ry1), 10, (255, 0, 0), -1)

        if (len(x_list) > 0) and (abs(x_list[-1] - (x + Rx1)) >= min_x_step):  # линии не могут быть слишком близко
            x_list.append(x + Rx1)
            # cv.circle(image, (100, y + Ry1), 10, (255, 0, 0), -1)
    if reverse_sort:
        x_list.sort(reverse=True)
    else:
        x_list.sort()
    # print("x_list", x_list)
    return x_list


def hsv_color_filter_cv(image, light_hsv=(93, 75, 74), dark_hsv=(151, 255, 255)):
    """
    Фильтрация цвета в диапазоне цветов HSV
    Параметры по умолчанию фильтруют синий цвет (заменяет белым)

    :param image:
    :param light_hsv:
    :param dark_hsv:
    :return: обработанное изображение
    """
    #
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(image_hsv, light_hsv, dark_hsv)
    #
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=2)
    #
    image[:, :, 0] = np.where(mask == 0, image[:, :, 0], 255)
    image[:, :, 1] = np.where(mask == 0, image[:, :, 1], 255)
    image[:, :, 2] = np.where(mask == 0, image[:, :, 2], 255)

    return image


def match_template_cv(img, template, templ_threshold=0.5, templ_metric=cv.TM_CCOEFF_NORMED):
    """
    Поиск шаблона на изображении методом opencv

    :param img: изображение
    :param template: шаблон
    :param templ_threshold: тресхолд
    :param templ_metric: метрика (параметр opencv)
    :return: изображение с найденными шаблонами, список bb (координат привязки шаблонов)
    """
    img_parsed = img.copy()
    #
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = template.shape[:2]
    #
    res = cv.matchTemplate(gray, template, templ_metric)
    loc = np.where(res >= templ_threshold)

    pt_list = []
    for pt in zip(*loc[::-1]):
        pt_list.append((pt[0], pt[1], pt[0] + w, pt[1] + h))
        cv.rectangle(img_parsed, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    print("  Найдено привязок темплейта (bb) на изображении: {}".format(len(pt_list)))
    return img_parsed, pt_list


def align_by_outer_frame_cv(img,
                            corner_regions,
                            diagonals_equality=0.02,
                            warped_shape=(0, 0),
                            canny_edges=False,
                            blur_kernel_shape=(5, 5),
                            horizontal_kernel_shape=(60, 1),
                            vertical_kernel_shape=(1, 60),
                            horizontal_iterations=2,
                            vertical_iterations=2,
                            coord_correction=False,
                            verbose=False):
    """
    Поиск прямоугольника по 4-м внешним углам и его выравнивание (трансформация)

    :param img: изображение для обработки
    :param corner_regions: список координат для поиска углов
    :param diagonals_equality: метрика относительного равенства диагоналей
    :param warped_shape: размер трансформируемой части
    :param canny_edges: производить преобразование Canny перед поиском углов
    :param blur_kernel_shape: размер ядра для функции блюра
    :param horizontal_kernel_shape: размер ядра для горизонталей
    :param horizontal_iterations: итераций для поиска горизонталей
    :param vertical_kernel_shape: размер ядра для вертикалей
    :param vertical_iterations: итераций для поиска вертикалей
    :param coord_correction: пробовать исправлять координаты
    :param verbose: печатать сопроводительные сообщения
    :return: image_out: обработанное изображение
             image_parsed: оригинальное изображение с пометками обработки
             transformed: если трансформация удачна, то True
             кортеж: 4 точки по которым делали трансформацию
    """
    # Копия исходного изображения
    image = img.copy()
    # Изображение для трансформации
    image_orig = img.copy()

    # ПРЕДОБРАБОТКА: удаление синего цвета
    light_blue = (93, 75, 74)
    dark_blue = (151, 255, 255)
    image = hsv_color_filter_cv(image.copy(), light_hsv=light_blue, dark_hsv=dark_blue)
    # show_image_cv(image, title="image")

    # ПРЕДОБРАБОТКА: переход к контурам Canny
    if canny_edges:
        print("  Задана предобработка Canny edges")
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = auto_canny(gray)
        kernel = np.ones((3, 3), np.uint8)
        gray = cv.dilate(gray, kernel, iterations=2)
        gray = cv.erode(gray, kernel, iterations=1)
        image = 255 - gray
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # show_image_cv(img_resize_cv(image, 700))
    # show_image_cv(image)

    # ##### ПО ГОРИЗОНТАЛИ ######
    #
    # ЛЕВЫЙ ВЕРХНИЙ УГОЛ
    # Поиск горизонтальных линий
    y_list = find_horizontal_cv(image,
                                corner_regions[0],
                                blur_kernel_shape=blur_kernel_shape,
                                horizontal_kernel_shape=horizontal_kernel_shape,
                                horizontal_iterations=horizontal_iterations,
                                min_y_step=10)
    if verbose:
        print("  ЛЕВЫЙ ВЕРХНИЙ y_list:", y_list)
    if len(y_list) > 0:
        Y1 = y_list[0]
    else:
        Y1 = -1

    # ПРАВЫЙ ВЕРХНИЙ УГОЛ
    # Поиск горизонтальных линий
    y_list = find_horizontal_cv(image,
                                corner_regions[1],
                                blur_kernel_shape=blur_kernel_shape,
                                horizontal_kernel_shape=horizontal_kernel_shape,
                                horizontal_iterations=horizontal_iterations,
                                min_y_step=10)
    if verbose:
        print("  ПРАВЫЙ ВЕРХНИЙ y_list:", y_list)
    if len(y_list) > 0:
        Y2 = y_list[0]
    else:
        Y2 = -1

    # ЛЕВЫЙ НИЖНИЙ УГОЛ
    # Поиск горизонтальных линий
    y_list = find_horizontal_cv(image,
                                corner_regions[2],
                                blur_kernel_shape=blur_kernel_shape,
                                horizontal_kernel_shape=horizontal_kernel_shape,
                                horizontal_iterations=horizontal_iterations,
                                min_y_step=30,
                                reverse_sort=True)
    if verbose:
        print("  ЛЕВЫЙ НИЖНИЙ y_list:", y_list)
    if len(y_list) > 0:
        Y3 = y_list[0]
    else:
        Y3 = -1

    # ПРАВЫЙ НИЖНИЙ УГОЛ
    # Поиск горизонтальных линий
    y_list = find_horizontal_cv(image,
                                corner_regions[3],
                                blur_kernel_shape=blur_kernel_shape,
                                horizontal_kernel_shape=horizontal_kernel_shape,
                                horizontal_iterations=horizontal_iterations,
                                min_y_step=30,
                                reverse_sort=True)
    if verbose:
        print("  ПРАВЫЙ НИЖНИЙ y_list:", y_list)
    if len(y_list) > 0:
        Y4 = y_list[0]
    else:
        Y4 = -1

    # ##### ПО ВЕРТИКАЛИ ######
    #
    # ЛЕВЫЙ ВЕРХНИЙ УГОЛ
    # Поиск вертикальных линий
    x_list = find_vertical_cv(image,
                              corner_regions[4],
                              blur_kernel_shape=blur_kernel_shape,
                              vertical_kernel_shape=vertical_kernel_shape,
                              vertical_iterations=vertical_iterations,
                              min_x_step=10)
    if verbose:
        print("  ЛЕВЫЙ ВЕРХНИЙ x_list:", x_list)
    if len(x_list) > 0:
        X1 = x_list[0]
    else:
        X1 = -1

    # ПРАВЫЙ ВЕРХНИЙ УГОЛ
    # Поиск вертикальных линий
    x_list = find_vertical_cv(image,
                              corner_regions[5],
                              blur_kernel_shape=blur_kernel_shape,
                              vertical_kernel_shape=vertical_kernel_shape,
                              vertical_iterations=vertical_iterations,
                              min_x_step=10,
                              reverse_sort=True)
    if verbose:
        print("  ПРАВЫЙ ВЕРХНИЙ x_list:", x_list)
    if len(x_list) > 0:
        X2 = x_list[0]
    else:
        X2 = -1

    # ЛЕВЫЙ НИЖНИЙ УГОЛ
    # Поиск вертикальных линий
    x_list = find_vertical_cv(image,
                              corner_regions[6],
                              blur_kernel_shape=blur_kernel_shape,
                              vertical_kernel_shape=vertical_kernel_shape,
                              vertical_iterations=vertical_iterations,
                              min_x_step=10)
    if verbose:
        print("  ЛЕВЫЙ НИЖНИЙ x_list:", x_list)
    if len(x_list) > 0:
        X3 = x_list[0]
    else:
        X3 = -1

    # ПРАВЫЙ НИЖНИЙ УГОЛ
    # Поиск вертикальных линий
    x_list = find_vertical_cv(image,
                              corner_regions[7],
                              blur_kernel_shape=blur_kernel_shape,
                              vertical_kernel_shape=vertical_kernel_shape,
                              vertical_iterations=vertical_iterations,
                              min_x_step=10,
                              reverse_sort=True)
    if verbose:
        print("  ПРАВЫЙ НИЖНИЙ x_list:", x_list)
    if len(x_list) > 0:
        X4 = x_list[0]
    else:
        X4 = -1

    # Получили КООРДИНАТЫ УГЛОВ
    if verbose:
        print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
    # Обозначим полученные углы
    if X1 != -1 and Y1 != -1:
        cv.circle(image, (X1, Y1), 10, (255, 0, 255), -1)
    if X2 != -1 and Y2 != -1:
        cv.circle(image, (X2, Y2), 10, (255, 0, 255), -1)
    if X3 != -1 and Y3 != -1:
        cv.circle(image, (X3, Y3), 10, (255, 0, 255), -1)
    if X4 != -1 and Y4 != -1:
        cv.circle(image, (X4, Y4), 10, (255, 0, 255), -1)

    # ##### ВОССТАНОВЛЕНИЕ ОТДЕЛЬНЫХ КООРДИНАТ #####
    if -1 in [X1, Y1, X2, Y2, X3, Y3, X4, Y4]:
        print("Отсутствует минимум одна координата")

    # Восстановление ОДНОЙ отсутствующей КООРДИНАТЫ из [X1, Y1, X2, Y2, X3, Y3, X4, Y4]
    if (X1 == -1) and (-1 not in [Y1, X2, Y2, X3, Y3, X4, Y4]):
        print("Нет одной координаты X1, исправляем")
        X1 = X3 + (X2 - X4)
        print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))

    # Восстановление ОДНОЙ отсутствующей КООРДИНАТЫ из [X1, Y1, X2, Y2, X3, Y3, X4, Y4]
    if (X2 == -1) and (-1 not in [X1, Y1, Y2, X3, Y3, X4, Y4]):
        print("Нет одной координаты X2, исправляем")
        X2 = X4 + (X1 - X3)
        print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
    #
    if (X3 == -1) and (-1 not in [X1, Y1, X2, Y2, Y3, X4, Y4]):
        print("Нет одной координаты X3, исправляем")
        X3 = X1 + (X4 - X2)
        print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
    #
    if (X4 == -1) and (-1 not in [X1, Y1, X2, Y2, X3, Y3, Y4]):
        print("Нет одной координаты X4, исправляем")
        X4 = X2 - (X1 - X3)
        print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
    #
    if (Y1 == -1) and (-1 not in [X1, X2, Y2, X3, Y3, X4, Y4]):
        print("Нет одной координаты Y1, исправляем")
        Y1 = Y2 - (Y4 - Y3)
        print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
    #
    if (Y2 == -1) and (-1 not in [X1, Y1, X2, X3, Y3, X4, Y4]):
        print("Нет одной координаты Y2, исправляем")
        Y2 = Y1 + (Y4 - Y3)
        print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
    #
    if (Y3 == -1) and (-1 not in [X1, Y1, X2, Y2, X3, X4, Y4]):
        print("Нет одной координаты Y3, исправляем")
        Y3 = Y4 - (Y2 - Y1)
        print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
    #
    if (Y4 == -1) and (-1 not in [X1, Y1, X2, Y2, X3, Y3, X4]):
        print("Нет одной координаты Y4, исправляем")
        Y4 = Y3 + (Y2 - Y1)
        print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))

    # TODO: Возможно восстановление любого отдельного Y и X отсутствующих одновременно

    # Если нет хотя бы одного угла, то трансформация невозможна
    if -1 in [X1, Y1, X2, Y2, X3, Y3, X4, Y4]:
        if verbose:
            print("Трансформация не возможна (не найден как минимум один угол)")
        # Выходим из функции
        return None, image, False, None

    # ##### ИСПРАВЛЕНИЕ ОТДЕЛЬНЫХ КООРДИНАТ #####
    if coord_correction:
        # Загрубляем метрику diagonals_equality
        prev_diagonals_equality = diagonals_equality
        diagonals_equality = 0.02
        print("Загрубляем метрику diagonals_equality: {} -> {}".format(prev_diagonals_equality, diagonals_equality))

        # Вычисляем длины диагоналей четырёхугольника по полученным точкам
        d1 = get_distance_pp((X1, Y1), (X4, Y4))
        d2 = get_distance_pp((X2, Y2), (X3, Y3))
        # Если диагонали разные, то пробуем исправить
        if not rel_equal(d1, d2, diagonals_equality):
            print("Диагонали разные: {}, {}".format(round(d1), round(d2)))
            # Вычисляем стороны
            # s1 = u.get_distance_pp((X1, Y1), (X2, Y2))
            # s2 = u.get_distance_pp((X2, Y2), (X3, Y3))
            # s3 = u.get_distance_pp((X3, Y3), (X4, Y4))
            # s4 = u.get_distance_pp((X4, Y4), (X1, Y1))

            # Восстановление ОДНОЙ НЕПРАВИЛЬНОЙ КООРДИНАТЫ из Y3 или Y4 при условии что остальные корректны
            # print(Y1, Y2, Y3, Y4)
            # print(X1, X2, X3, X4)
            print("Пробуем исправить один нижний угол, при условии что остальные корректны")
            dY = abs(Y1 - Y2)
            if Y1 > Y2:
                print("Y3:", Y3, "->", Y4 + dY)
                Y3 = Y4 + dY
                cv.circle(image, (X3, Y3), 10, (0, 0, 255), -1)
            else:
                print("Y4:", Y4, "->", Y3 + dY)
                Y4 = Y3 + dY
                cv.circle(image, (X4, Y4), 10, (0, 0, 255), -1)

    # Вычисляем длины диагоналей четырёхугольника по полученным точкам
    d1 = get_distance_pp((X1, Y1), (X4, Y4))
    d2 = get_distance_pp((X2, Y2), (X3, Y3))
    print("Диагонали прямоугольника из найденных координат: {}, {}.".format(round(d1), round(d2)))
    # Если диагонали разные, то трансформация будет некорректна
    if not rel_equal(d1, d2, diagonals_equality):
        print("Трансформация не получается (разные диагонали), d1/d2=", round(min(d1, d2) / max(d1, d2), 3))
        # Выходим из функции
        return None, image, False, None

    # #############################################################
    # HOOK: Сдвиг на 5 пикселей чтобы не исчезала рамка справа и снизу
    X1, Y1, X2, Y2, X3, Y3, X4, Y4 = X1, Y1, X2 + 5, Y2, X3, Y3 + 5, X4 + 5, Y4 + 5
    # #############################################################

    # ТРАНСФОРМАЦИЯ
    points = np.array([(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)])
    warped = perspective.four_point_transform(image_orig, points)
    if verbose and warped is not None:
        print("Трансформация перспективы выполнена.")
    # print("image_orig.shape", image_orig.shape)
    # print("warped.shape", warped.shape)

    # Приведем размер warped к единообразному размеру
    X_b, Y_b = 0, 0
    if 0 not in warped_shape:
        warped = cv.resize(warped, warped_shape, interpolation=cv.INTER_AREA)
        # Смещение чтобы разместить warped посередине image_out
        X_b = int((image_orig.shape[1] - warped.shape[1]) / 2)
        Y_b = int((image_orig.shape[0] - warped.shape[0]) / 2)
    elif warped_shape[0] != 0 and warped_shape[1] == 0:
        warped_shape = (warped_shape[0], int(((Y3 - Y1) + (Y4 - Y2)) / 2))
        warped = cv.resize(warped, warped_shape, interpolation=cv.INTER_AREA)
        #
        X_b = int((image_orig.shape[1] - warped.shape[1]) / 2)
        Y_b = 65
    # print("warped.shape after resize", warped.shape)
    # print("X_b, Y_b", X_b, Y_b)

    # Переносим трансформированное изображение на выходную картинку
    image_out = (np.ones((image_orig.shape[0], image_orig.shape[1], 3)) * 255).astype('uint8')
    Xdest1, Ydest1 = X_b, Y_b
    Xdest2, Ydest2 = X_b + warped.shape[1], Y_b + warped.shape[0]

    # warped + Y_b может быть больше image_orig.shape[0]
    Y_cut = Y_b + warped.shape[0] - image_orig.shape[0]
    # print("Y_cut", Y_cut)
    if Y_cut > 0:
        image_out[Ydest1:Ydest2, Xdest1:Xdest2] = warped[:-Y_cut, :]
    else:
        image_out[Ydest1:Ydest2, Xdest1:Xdest2] = warped

    # print("image_out.shape", image_out.shape)
    # show_image_cv(img_resize_cv(image, img_size=500), title='image')
    # show_image_cv(img_resize_cv(image_out, img_size=500), title='image_out')

    # Возвращаем обработанное изображение, изображение с пометками, положительный статус операции,
    # координаты перенесенного на image прямоугольника warped.
    return image_out, image, True, (Xdest1, Ydest1, Xdest2, Ydest2)


# #############################################################
#                    ФУНКЦИИ МАТЕМАТИЧЕСКИЕ
# #############################################################
def get_distance_pp(p1, p2):
    """
    Вычисляет расстояние между двумя точками на плоскости
    :param p1: точка на плоскости tuple (x1, y1)
    :param p2: точка на плоскости tuple (x2, y2)
    :return: расстояние
    """
    distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return distance


def scale_to_01_range(x):
    """
    Scale and move the coordinates, so they fit [0; 1] range
    :param x: numpy array to process
    :return:
    """
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def rel_equal(x, y, e):
    """
    Возвращает истина, если числа совпадают с указанной относительной точностью
    :param x: число для сравнения
    :param y: числа для сравнения
    :param e: относительная точность
    :return: boolean
    """
    # print("rel_equality", x, y, e, "abs(abs(x / y) - 1) < e :", abs(abs(x / y) - 1) < e,
    #       "math.isclose(x, y, rel_tol=e) :", math.isclose(x, y, rel_tol=e))
    return math.isclose(x, y, rel_tol=e)


# #############################################################
#                      ФУНКЦИИ ПРОЧИЕ
# #############################################################
def clean_folder(path):
    """
    Очистка папки с удалением только файлов внутри нее
    :param path: путь к папке
    """
    files_to_remove = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files_to_remove.append(os.path.join(path, file))
    for file in files_to_remove:
        os.remove(file)


def get_files_by_type(source_files, type_list=None):
    """
    Получает список файлов и список типов (расширений)
    :param source_files: список файлов
    :param type_list: список типов (расширений файлов) или одно расширение как строка
    :return: список файлов, отобранный по типу (расширениям)
    """
    if type_list is None:
        type_list = []
    elif type(type_list) is str:
        type_list = [type_list]

    result_file_list = []
    for f in source_files:
        _, file_extension = os.path.splitext(f)
        if file_extension in type_list:
            result_file_list.append(f)
    return result_file_list


def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    :param n: number of distinct color
    :param name: argument name must be a standard mpl colormap name
    :return: function
    """
    return plt.cm.get_cmap(name, n)


def txt_separator(string, n=80, txt='', txt_align='center'):
    """
    Формирует строку из символов и текстовой надписи.
    Текст накладывается в начале, в середине или в конце строки
    :param string: строка или символ, повторением которых формируется разделитель
    :param n: количество символов в стоке (если 0, то попробовать подогнать по консоли)
    :param txt: текст для печати (если слишком длинный, то обрезается)
    :param txt_align: как выравнивать текст в строке (left, center, right)
    :return:
    """
    assert type(string) != 'str', "Ожидается тип данных 'str'"
    n = 0 if n < 0 else n      # отрицательное n не имеет смысла
    n = 128 if n > 128 else n  # слишком большое n не имеет смысла

    if n == 0:  # пробуем получить ширину консоли
        stty_size = os.popen('stty size', 'r').read().split()
        if len(stty_size) == 2:
            n = int(stty_size[1])
        else:  # если не получилось присваиваем стандартную ширину
            n = 80

    # Выходная строка без текста
    if txt == '':
        k = n // len(string) + 1  # сколько раз брать строку
        out_string = string * k   # выходная строка
        out_string = out_string[:n] if len(out_string) > n else out_string  # ограничиваем длину строки параметром n
        return out_string
    # Выходная строка с наложением текста
    else:
        if len(txt) == n:  # если текст длинной n, то возвращаем текст
            return txt
        elif len(txt) > n:  # если текст слишком длинный, то возвращаем текст, обрезая до длинны n
            txt = txt[:n]
            return txt
        else:
            k = n // len(string) + 1  # сколько раз брать строку
            out_string = string * k   # выходная строка
            out_string = out_string[:n] if len(out_string) > n else out_string  # ограничиваем длину строки параметром n
            if txt_align == 'left':
                out_string = txt + out_string[len(txt):]
                return out_string
            elif txt_align == 'right':
                out_string = out_string[:-len(txt)] + txt
                return out_string
            elif txt_align == 'center':
                start_txt_pos = (n - len(txt)) // 2
                out_string = out_string[:start_txt_pos] + txt + out_string[-start_txt_pos:]
                return out_string


def digits_amount(string):
    """
    Подсчитывает количество цифр в строке
    :param string: строка
    :return: количество цифр
    """
    return sum(map(lambda x: x.isdigit(), string))


def frequency_sort(items):
    """
    Сортировка списка по частоте.
    При одинаковой частоте элементы встанут по возрастанию
    :param items: список
    :return: сортированный список
    """
    temp = []
    sort_dict = {x: items.count(x) for x in items}
    for k, v in sorted(sort_dict.items(), key=lambda x: x[1], reverse=True):
        temp.extend([k] * v)
    return temp


def check_string(string, pattern):
    """
    Проверяет соответствие стоки регулярному выражению
    :param string: строка
    :param pattern: паттерн регулярного выражения
    :return: True если строка соответствует регулярному выражению,
        False - если не соответствует,
        None - если паттерн ошибочный
    """
    try:
        compiled_regex = re.compile(pattern)
        return bool(compiled_regex.fullmatch(string))
    except re.error:
        return None


# #############################################################
#                  ЭКСПЕРИМЕНТАЛЬНЫЕ функции
# #############################################################

# def preprocessing_cv(img, inversion=False):
#     """
#     Препроцессинг изображения: переход в ч/б, бинаризация и морфология
#
#     :param img: изображение
#     :param inversion: делать инверсию изображения
#     :return: обработанное ч/б изображение
#     """
#     img = img.copy()
#     #
#     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
#     # Бинаризация изображения
#     if inversion:
#         img_bin = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
#     else:
#         img_bin = cv.threshold(img, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
#
#     # Морфологические операции для улучшения качества изображения
#     kernel = np.ones((3, 3), np.uint8)
#     img_bin = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel)
#     img_bin = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
#
#     # show_image_cv(img_resize_cv(img_bin, 800), title="img preprocessed")
#     return img_bin


def preprocessing_bw_cv(img, force_threshold=True, keep_dimension=True):
    """
    Препроцессинг изображения: переход в ч/б, тресхолд опционально.
    Принимает 3-х канальное изображение.
    Возвращает 1-канальное или 3-х канальное изображение

    :param img: изображение
    :param force_threshold: тресхолд выходного изображение
    :param keep_dimension: сохранять количество каналов
    :return: обработанное ч/б изображение
    """
    img_bw = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)

    if force_threshold:
        img_bw = cv.GaussianBlur(img_bw, (5, 5), 0)
        thresh = cv.adaptiveThreshold(img_bw,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,2)

        # Применение морфологической операции расширения для утолщения линий
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv.dilate(thresh, kernel, iterations=1)
        thresh = cv.erode(thresh, kernel, iterations=1)

        # Инвертирование изображения обратно
        img_bw = cv.bitwise_not(thresh)

    # Сохраняем размерность
    if keep_dimension:
        img_bw = cv.cvtColor(img_bw, cv.COLOR_GRAY2BGR)

    # show_image_cv(img_resize_cv(img_bw, 800), title="img preprocessed bw")
    return img_bw


def get_custom_jpeg(img_gray, region_list):
    """
    Создание JPEG с переменным качеством.
    Основное изображение сжимается с большей компрессией.
    Регионы сжимаются с меньшей компрессией (лучшим качеством)

    :param img_gray: ч/б изображение в формате numpy
    :param region_list: список регионов
    :return: обработанное ч/б изображение в формате PILLOW
    """
    img_gray = Image.fromarray(img_gray, mode="L")

    # Сжимаем основное изображение
    buffered = io.BytesIO()
    img_gray.save(buffered, format="JPEG", subsampling=0, quality=s.JPEG_LOW_QUALITY)
    img_poor = Image.open(buffered)
    # img_poor.show()

    for r in region_list:
        # Пропускаем специальные QR коды
        # if r['qr_txt'] in s.QRS_SPECIAL_TXT:
        #     continue

        # Пропускаем QR коды без ключа boxes
        if 'boxes' not in r:
            continue

        X1 = r['qr_coord'][0]
        Y1 = r['qr_coord'][1]
        X2 = r['boxes'][2]
        Y2 = r['boxes'][3]
        # print("X1, Y1, X2, Y2", X1, Y1, X2, Y2)

        # Выбираем область изображения, которую оставить качественной
        box = (X1, Y1, X2, Y2)
        region = img_gray.crop(box)
        # region.show()
        # Сжимаем выбранную область минимально
        region_buffered = io.BytesIO()
        region.save(region_buffered, "JPEG", quality=s.JPEG_TOP_QUALITY)
        region_good = Image.open(region_buffered)
        # region_good.show()

        # Вставляем область в изображение
        img_poor.paste(region_good, box)

    # image_poor.save("out_files/poor.jpg", format="JPEG", subsampling=0, quality=s.JPEG_QUALITY)
    # img_poor.show()
    return img_poor


def image_is_close_to_white_cv(img, threshold=200, white_ratio=0.9):
    """
    Проверка близости изображения к белому цвету

    :param img: изображение для обработки
    :param threshold:
    :param white_ratio: доля белых пикселей на тресхолде
    :return: True если изображение близко белому, доля белых пикселей
    """
    image = img.copy()

    # Преобразование изображения в формат HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Определение границ для белого цвета в формате HSV
    lower_white = np.array([0, 0, threshold], dtype=np.uint8)
    upper_white = np.array([180, 55, 255], dtype=np.uint8)

    # Создание маски для белых областей
    white_mask = cv.inRange(hsv_image, lower_white, upper_white)

    # Подсчет количества белых пикселей
    white_pixels = cv.countNonZero(white_mask)

    # Подсчет общего количества пикселей
    total_pixels = image.shape[0] * image.shape[1]

    # Вычисление доли белых пикселей
    white_ratio_actual = white_pixels / total_pixels

    # Проверка, превышает ли доля белых пикселей заданный порог
    return white_ratio_actual >= white_ratio, white_ratio_actual


def find_document_by_grabcut_cv(img, max_size=1080):
    """
    Ищет документ на изображении методом GrabCut
    Хорошо убирает фон если бланк не залазит на края изображения

    :param img: изображение для обработки
    :param max_size: максимальное разрешение, в котором проводить обработку
    :return: True если обработка успешна, обработанное изображение
    """

    def order_points(pts):
        """Rearrange coordinates to order:
           top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)
        # Top-left point will have the smallest sum.
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point will have the largest sum.
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        # Top-right point will have the smallest difference.
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left will have the largest difference.
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect.astype('int').tolist()

    def find_dest(pts):
        (tl, tr, br, bl) = pts
        # Finding the maximum width.
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Finding the maximum height.
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # Final destination co-ordinates.
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
        return order_points(destination_corners)

    # Create a copy of original image for final transformation
    img_orig = img.copy()
    #
    H, W = img_orig.shape[:2]
    # print("  Размер оригинального изображения: {}".format((H, W)))

    # Resize image to workable size
    max_dim = max(img.shape)
    if max_dim > max_size:
        resize_scale = max_size / max_dim
        img = cv.resize(img, None, fx=resize_scale, fy=resize_scale)
    #
    h, w = img.shape[:2]
    # print("  Размер изображения после ресайза: {}".format((h, w)))

    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=3)

    # Check image is almost white
    if image_is_close_to_white_cv(img, threshold=200, white_ratio=0.9)[0]:
        print("  Изображение близко к белому, выход")
        return False, None

    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # rect = (20, 20, img.shape[1] - 40, img.shape[0] - 40)
    rect = (15, 15, img.shape[1] - 30, img.shape[0] - 30)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    # Gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11, 11), 0)

    # Edge Detection.
    canny = cv.Canny(gray, 0, 200)
    canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    # If there weren't any contours
    if len(page) == 0:
        print("  Не найдено хотя бы 1-го контура, выход")
        return False, None

    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    corners = []
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv.arcLength(c, True)
        corners = cv.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # print("  Найдено углов: {}".format(len(corners)))

    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())

    # For 4 corner points being detected.
    if len(corners) >= 4:
        corners = order_points(corners)
        # print("  corners", corners)

        # Recalculating corners in accordance with original image size
        for corner in corners:
            corner[0] = int(corner[0] / h * H)
            corner[1] = int(corner[1] / w * W)
        # print("  corners after recalculating", corners)

        # Отношение диагоналей и размеры четырехугольника из найденных углов
        X1 = corners[0][0]
        Y1 = corners[0][1]
        X2 = corners[1][0]
        Y2 = corners[1][1]
        X3 = corners[2][0]
        Y3 = corners[2][1]
        X4 = corners[3][0]
        Y4 = corners[3][1]
        #
        diag_ratio = get_distance_pp((X1, Y1), (X3, Y3)) / get_distance_pp((X2, Y2), (X4, Y4))
        if diag_ratio > 1.0:
            diag_ratio = 1.0 / diag_ratio
        #
        if X2 - X1 < W * 0.5 or Y4 - Y1 < H * 0.5 or diag_ratio < 0.85:
            print("  Cлишком маленький или искаженный объект: {} * {}".format(X2 - X1, Y4 - Y1))
            return False, None

        # Draw points
        # point_count = 0
        # for corner in corners:
        #     cv.circle(img_parsed, corner, 10, s.blue, -1)
        #     point_count += 1
        #     cv.putText(img_parsed, str(point_count), corner, cv.FONT_HERSHEY_SIMPLEX, 2, s.green, 2)
        # show_image_cv(img_resize_cv(img_parsed, img_size=900), title='img_parsed')

        #
        destination_corners = find_dest(corners)
        # print("  destination_corners", destination_corners)

        # Getting the homography.
        M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

        # Perspective transform using homography.
        img_out = cv.warpPerspective(img_orig, M,
                                     (destination_corners[2][0], destination_corners[2][1]),
                                     flags=cv.INTER_LINEAR)

        # Resize результат к исходному изображению
        img_out = cv.resize(img_out, (W, H), interpolation=cv.INTER_AREA)
        # show_image_cv(img_resize_cv(img_out, img_size=900), title='img_perspective ' + str(img_out.shape))

        return True, img_out
    else:
        print("  Не найдено хотя бы 4-ре угла, выход")
        return False, None


def four_point_transform(image, pts):

    def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        pts = np.array(pts)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped
