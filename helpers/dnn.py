"""
Функции, связанные с вызовом моделей
и обработкой результатов модулем OpenCV DNN
"""
# import numpy as np
import cv2 as cv
import time


# #########################################################
# Функции для работы с моделями DNN opencv - универсальные
# #########################################################
def get_model_dnn(model_file, force_cuda=False, verbose=False):
    """
    Загрузка модели из указанного файла
    :param model_file: путь к модели для загрузки
    :param force_cuda: использовать CUDA
    :param verbose: выводить дополнительную информацию
    :return: model: загруженная модель
    """
    if verbose:
        print('Загружаем модель: {}'.format(model_file))
    time_0 = time.perf_counter()
    #
    model = cv.dnn.readNet(model_file)
    if force_cuda:
        print("  Try to use CUDA")
        model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("  Running on CPU")
        model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    #
    time_1 = time.perf_counter()
    if verbose:
        print('  Время загрузки модели, с: {:.2f}'.format(time_1 - time_0))
    return model


def get_blob_dnn(source_img, input_shape=(640, 640), to_gray=False, subtract_mean=False):
    """
    Подготовка изображения для передачи в модель

    :param source_img: исходное изображение
    :param input_shape: размер к которому приводить изображение
    :param to_gray: переходить к ч/б
    :param subtract_mean: центрировать без скейлинга
    :return: обработанное изображение
    """
    # ресайз к img_size, scale factor = 1, subtract mean BGR
    # result = cv.dnn.blobFromImage(source_img, 1, input_shape, (104, 117, 123))

    if to_gray:
        # переход к ч/б, ресайз к img_size и нормализация
        gray = cv.cvtColor(source_img, cv.COLOR_BGR2GRAY)
        # gray = cv.threshold(grey, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        result = cv.dnn.blobFromImage(gray, 1 / 255.0, input_shape)
        return result

    if subtract_mean:
        # ресайз к img_size, scale factor = 1, subtract mean BGR
        result = cv.dnn.blobFromImage(source_img, 1, input_shape, (104, 117, 123))  # BGR
        # result = cv.dnn.blobFromImage(source_img, 1, input_shape, (123, 117, 103))  # RGB
        return result

    # ресайз к img_size, нормализация и замена между собой R и B каналов (переход к RGB)
    result = cv.dnn.blobFromImage(source_img, 1 / 255.0, input_shape, swapRB=True)
    return result


def get_pred_dnn(model, blob):
    """
    Получение предикта из модели и подготовленного изображения

    :param model: модель
    :param blob: подготовленное изображение
    :return: обработанное изображение
    """
    model.setInput(blob)
    predictions = model.forward()
    return predictions
