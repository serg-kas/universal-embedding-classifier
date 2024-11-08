"""
Основной модуль программы.
Режимы работы см.operation_mode_list
"""
# import numpy as np
import cv2 as cv
from PIL import Image
#
import os
import sys
import time
from datetime import datetime
# import shutil
import requests
# import magic
# import threading
import io
import json
import base64
from pprint import pprint

#
import config
import log
import settings as s
import tools as t
import helpers.classify_cnnfe as cnnfe
import helpers.dnn as dnn
import helpers.utils as u


# Рабочая директория: полный путь и локальная папка
working_folder_full_path = os.getcwd()
# working_folder_base_name = os.path.basename(working_folder_full_path)
print("Рабочая директория (полный путь): {}".format(working_folder_full_path))


# ##################### РЕЖИМЫ РАБОТЫ #########################
operation_mode_list = [
                       'rebuild_emb',
                       'cnnfe_classifier',
                       ]
default_mode = operation_mode_list[-1]  # режим работы по умолчанию


def process(operation_mode, source_files, out_path):
    """
    Функция запуска рабочего процесса по выбранному
    режиму работы
    :param operation_mode: режим работы
    :param source_files: список файлов для обработки
    :param out_path: путь для сохранения результата
    :return:
    """
    time_start = time.time()

    # Выбор режима работы. Находит первый подходящий режим.
    # (например, по параметру 'emB' будет выбран режим 'rebuild_emb')
    for curr_mode in operation_mode_list:
        if operation_mode.lower() in curr_mode:
            operation_mode = curr_mode
            break
    print('Режим работы, заданный в параметрах командной строки: {}'.format(operation_mode))



    # ##################### rebuild_emb #######################
    # Пересчитываются и сохраняются эмбеддинги датасета.
    # Работает в датасете EMB_PATH_HANDLE
    # #########################################################
    if operation_mode == 'rebuild_emb':
        # Обрабатывать папку
        print("Работаем в датасете {}".format(s.EMB_PATH_HANDLE))
        assert os.path.isdir(s.EMB_PATH_HANDLE), "Датасет {} не найден".format(s.EMB_PATH_HANDLE)

        # True - создать скаттер, не пересчитывая эмбеддинги
        prep_scatter_ONLY = s.PREP_SCATTER_ONLY
        # Пересчитывать эмбеддинги выборочно, только в указанных папках (если список пуст, то пересчитывать всё)
        folders_to_process = s.FOLDERS_TO_PROCESS

        # Список папок для обработки
        data_folders = []  # полные пути к папкам
        labels = []        # имена папок - это метки классов
        for f in sorted(os.listdir(s.EMB_PATH_HANDLE)):
            if os.path.isdir(os.path.join(s.EMB_PATH_HANDLE, f)):
                if (f in folders_to_process) or (len(folders_to_process) == 0):
                    print("  подключаем папку: {}".format(f))
                    data_folders.append(os.path.join(s.EMB_PATH_HANDLE, f))
                    labels.append(f)
        print("Всего папок для обработки: {}".format(len(data_folders)))
        #
        if not prep_scatter_ONLY:
            # Получаем модель
            model_fe = dnn.get_model_dnn(s.MODEL_DNN_FILE_fe, force_cuda=s.FORCE_CUDA_fe, verbose=s.VERBOSE)

            # Получаем функцию предобработки изображения
            preprocess_function = None
            if s.FORCE_PREPROCESS_IMG:
                # preprocess_function = u.autocontrast_cv
                # preprocess_function = u.sharpen_image_cv
                preprocess_function = u.preprocessing_bw_cv

            # Создаем эмбеддинги всего датасета
            cnnfe.create_embeddings(model_fe, data_folders, preprocess=True, preprocess_func=preprocess_function)

        # Пройдем по сохраненным эмбеддингам и соберем их в один массив
        all_embeddings, class_pointers = cnnfe.collect_embeddings(data_folders)
        # Создадим скаттер
        cnnfe.create_tsne_scatter(labels, all_embeddings, class_pointers)

    # ################### cnnfe_classifier ####################
    # Универсальный CNNfe классификатор.
    # Работает в датасете EMB_PATH_HANDLE
    # Сохраняет результат под именем, соответствующим классу.
    # #########################################################
    if operation_mode == 'cnnfe_classifier':
        # Обрабатывать папку (по умолчанию из settings.py)
        assert os.path.isdir(s.EMB_PATH_HANDLE), "Не найден датасет {}".format(s.EMB_PATH_HANDLE)
        print("Работаем в датасете {}".format(s.EMB_PATH_HANDLE))
        #
        class_folders_list = []  # если список пуст, то будут загружены все папки с данными

        # Загрузим папки, метки классов, массив эмбеддигов, указатели классов
        data_folders, labels, all_embeddings, class_pointers = cnnfe.cnnfe_preparation(s.EMB_PATH_HANDLE,
                                                                                       class_folders_list)
        # Получаем модель
        model_fe = dnn.get_model_dnn(s.MODEL_DNN_FILE_fe, force_cuda=s.FORCE_CUDA_fe, verbose=s.VERBOSE)

        # Получаем функцию предобработки изображения
        preprocess_function = None
        if s.FORCE_PREPROCESS_IMG:
            # preprocess_function = u.autocontrast_cv
            # preprocess_function = u.sharpen_image_cv
            preprocess_function = u.preprocessing_bw_cv

        # Загружаем только изображения
        img_file_list = u.get_files_by_type(source_files, s.ALLOWED_IMAGES)
        if len(img_file_list) < 1:
            print("Не нашли изображений для обработки")

        img_list = w.get_images(img_file_list, autorotate=None, verbose=s.VERBOSE)
        # img_list = w.get_images(img_file_list, autorotate='TESS', verbose=s.VERBOSE)

        counter = 0
        for img in img_list:
            # Вызов функции препроцессинга изображения
            if s.FORCE_PREPROCESS_IMG:
                print("Используется функция предобработки изображения: {}".format(preprocess_function.__name__))
                img = preprocess_function(img)
            #
            pred_final, _, img_parsed, img_scatter = cnnfe.cnnfe_classifier(model_fe,
                                                                            s.N_votes_fe,
                                                                            labels,
                                                                            all_embeddings,
                                                                            class_pointers,
                                                                            img,
                                                                            put_txt=True,
                                                                            tsne=False,
                                                                            verbose=s.VERBOSE)
            #
            if img_scatter is not None:
                Image.fromarray(img_scatter[:, :, ::-1]).show()

            out_file_name = os.path.join(out_path, 'pred_' + pred_final + '_(' + str(counter) + ')' + '.png')
            cv.imwrite(out_file_name, img_parsed)
            counter += 1






if __name__ == '__main__':
    """
    В программу можно передать параметры командной строки:
    sys.argv[1] - operation_mode - режим работы 
    sys.argv[2] - source - путь к папке или отдельному файлу для обработки
    sys.argv[3] - out_path - путь к папке для сохранения результатов
    """

    # Проверим наличие и создадим рабочие папки если их нет
    config.check_folders([s.SOURCE_PATH, s.OUT_PATH, s.MODELS_PATH, s.EMB_PATH],
                         verbose=s.VERBOSE)

    # Параметры командной строки
    OPERATION_MODE = default_mode if len(sys.argv) <= 1 else sys.argv[1]
    SOURCE = s.SOURCE_PATH if len(sys.argv) <= 2 else sys.argv[2]
    OUT_PATH = s.OUT_PATH if len(sys.argv) <= 3 else sys.argv[3]

    # Если в параметрах источник - это папка
    if os.path.isdir(SOURCE):
        SOURCE_FILES = []
        for file in sorted(os.listdir(SOURCE)):
            if os.path.isfile(os.path.join(SOURCE, file)):
                _, file_extension = os.path.splitext(file)
                # Берем только разрешенные типы файлы
                if file_extension in s.ALLOWED_TYPES:
                    SOURCE_FILES.append(os.path.join(SOURCE, file))
        # Отправляем в работу не проверяя, что source_files может быть пуст
        process(OPERATION_MODE, SOURCE_FILES, OUT_PATH)

    # Если в параметрах источник - это файл
    elif os.path.isfile(SOURCE):
        _, file_extension = os.path.splitext(SOURCE)
        # Берем только разрешенный типы файлов
        if file_extension in s.ALLOWED_TYPES:
            source_file = SOURCE
            print('Обрабатываем файл: {}'.format(source_file))
            # Отправляем файл в работу
            process(OPERATION_MODE, [source_file], OUT_PATH)

    # Иначе не нашли данных для обработки
    else:
        print("Не нашли данных для обработки: {}".format(SOURCE))
