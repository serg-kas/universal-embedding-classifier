"""
Модуль проверки и настройки конфигурации.
Проверит наличие и при необходимости создаст рабочие папки.
Проверит наличие и при необходимости загрузит файл(ы) моделей.
"""
import os
import requests
from tqdm import tqdm
#
import settings as s


def check_folders(folders_to_check, verbose=False):
    """
    Проверяет наличие и при необходимости создает рабочие папки

    :param folders_to_check: список папок для проверки
    :param verbose: выводить сообщения
    :return: список созданных папок
    """
    if verbose:
        print("Проверяем наличие рабочих папок: {}".format(folders_to_check))
    #
    folders_created = []
    for folder_path in folders_to_check:
        if not (folder_path in os.listdir('.')):
            os.mkdir(folder_path)
            if verbose:
                print("  Создали отсутствующую папку: '{}'".format(folder_path))
            folders_created.append(folder_path)
    return folders_created


def check_files(files_to_check, verbose=False):
    """
    Проверяет наличие и при необходимости скачивает файлы

    :param files_to_check: список файлов для проверки
    :param verbose: выводить сообщения
    :return: список загруженных файлов моделей
    """
    if verbose:
        print("Проверяем наличие необходимых файлов: {}".format(files_to_check))
    #
    files_downloaded = []
    for file_path in files_to_check:
        if not os.path.isfile(file_path):
            if verbose:
                print("Отсутствует файл: {}".format(file_path))
            if file_path in s.URL_files_dict:
                download_url = s.URL_files_dict[file_path]
                #
                print("  Пробуем загрузить по ссылке: {} ...".format(download_url[:64]))
                response = requests.get(download_url)
                # with open(file_path, 'wb') as file:
                #     file.write(response.content)

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # размер блока в байтах
                with open(file_path, 'wb') as file, tqdm(
                        desc=file_path,
                        total=total_size,
                        unit='iB',
                        unit_scale=True) as bar:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        bar.update(len(data))

                print("  Загрузили отсутствующий файл: '{}'".format(file_path))
            else:
                if verbose:
                    print("  В настройках нет ссылки для загрузки данного файла")
    return files_downloaded

