"""
Модуль работы с настройками
"""
import os
import json
from dotenv import load_dotenv

# При наличии файла env загружаем из него переменные окружения
dotenv_path = os.path.join(os.path.dirname(__file__), 'env')   #.env
if os.path.exists(dotenv_path):
    print("Загружаем переменные окружения из файла: {}".format(dotenv_path))
    load_dotenv(dotenv_path)


def get_variable_name(variable):
    """
    Возвращает имя переменной как строку.
    Если переменная по имени не найдена, возвращает None

    :param variable: переменная
    :return: имя переменной
    """
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None


def get_value_from_env(variable, default_value=None, prefix_name="APP_", verbose=True):
    """
    Ищет значение в переменных окружения.
    Если параметр variable - переменная и есть соответствующее значение в переменных окружения,
    то возвращает это значение. Если значения нет, то возвращает значение переменной variable.

    Если variable - имя переменной (строка) и есть соответствующее значение
    в переменных окружения, то возвращает это значение. Если значения нет, то возвращает default_value

    :param variable: существующая переменная или имя переменной (строка)
    :param default_value: значение по умолчанию
    :param prefix_name: префикс прибавляется к имени переменной
    :param verbose: выводить подробные сообщения
    :return: значение переменной
    """
    variable_name = get_variable_name(variable)
    if variable_name != 'variable':
        got_variable = True
    else:
        got_variable = False
        variable_name = variable

    # Переменная ищется с префиксом в верхнем регистре
    value = os.getenv(prefix_name + variable_name.upper())
    if value is not None:
        if type(default_value) is bool:
            value = bool(value)
        elif type(default_value) is int:
            value = int(value)
        elif type(default_value) is float:
            value = float(value)
        elif type(default_value) is list:
            try:
                value = json.loads(value)
            except ValueError as e:
                # При неудачном преобразовании в json останется тип str
                if verbose:
                    print("  Ошибка: {}".format(e))
                print("  Не удалось прочитать как список: {}".format(value))
        #
        if verbose:
            print("  Получили значение из переменной окружения: {}={}".format(variable_name, value))
            # print(variable_name, value, type(value)))
        return value
    else:
        if got_variable:
            if verbose:
                print("  Не найдено значения переменной {} в переменных окружения, "
                      "оставлено без изменения: {}".format(variable_name, variable))
            return variable
        else:
            if verbose:
                print("  Не найдено значения переменной {} в переменных окружения, "
                      "по умолчанию: {}".format(variable_name, default_value))
            return default_value


# #############################################################
#                       ОБЩИЕ ПАРАМЕТРЫ
# #############################################################

# Флаг вывода подробных сообщений в консоль (уровень logging.DEBUG)
VERBOSE = get_value_from_env("VERBOSE", default_value=False)
CONS_COLUMNS = 0  # ширина консоли (0 - попытаться определить автоматически)

# Папки по умолчанию
SOURCE_PATH = 'source_files'
OUT_PATH = 'out_files'
MODELS_PATH = 'models'
EMB_PATH = get_value_from_env("EMB_PATH", default_value='data')  # общая папка для датасетов/эмбеддингов

# Допустимые форматы изображений для загрузки в программу
ALLOWED_IMAGES = ['.jpg', '.jpeg', '.png']
ALLOWED_TYPES = ALLOWED_IMAGES

# Словарь для хранения ссылок на необходимые файлы
URL_files_dict = {}


# #############################################################
#                     ПАРАМЕТРЫ МОДЕЛЕЙ
# #############################################################

# Модель CNN экстрактора фич для вызова из opencv
MODEL_DNN_FILE_fe = 'models/vgg16fe.onnx'
MODEL_DNN_FILE_fe_URL = get_value_from_env("MODEL_DNN_FILE_FE_URL", default_value='')
URL_files_dict[MODEL_DNN_FILE_fe] = MODEL_DNN_FILE_fe_URL

#
FORCE_CUDA_fe = False
INPUT_HEIGHT_fe = 224
INPUT_WIDTH_fe = 224

# Ансамблевая обработка
N_votes_fe = 10
CONFIDENCE_THRESHOLD_votes = 0.40

# Папки с данными/эмбеддингами внутри папки EMB_PATH
CNNFE_DATA_default = os.path.join(EMB_PATH, "masks")

# Использовать данный датасет (применяется для указания где производить классификацию)
EMB_PATH_HANDLE = get_value_from_env("EMB_PATH_HANDLE", default_value=CNNFE_DATA_default)

# Собирать результаты предиктов классификатора в новый датасет
CNNFE_DATA_COLLECT = get_value_from_env("CNNFE_DATA_COLLECT", default_value=False)
CNNFE_DATA_checkboxes_collect = os.path.join(EMB_PATH, "masks_collect")  # папки для сохранения собранных данных
CNNFE_DATA_COLLECT_limit = 0  # 0 = не ограничивать количество

# Параметры режима пересчета эмбеддингов (rebuild_emb)
FORCE_PREPROCESS_IMG = get_value_from_env("FORCE_PREPROCESS_IMG", default_value=False, verbose=False)
FOLDERS_TO_PROCESS = get_value_from_env("FOLDERS_TO_PROCESS", default_value=[], verbose=False)


# #############################################################
#                           ПРОЧЕЕ
# #############################################################

# ####################### Цвета RGB #######################
black = (0, 0, 0)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
purple = (255, 0, 255)
turquoise = (255, 255, 0)
white = (255, 255, 255)

# ################### Цвета для консоли ###################
BLACK_cons = '\033[30m'
RED_cons = '\033[31m'
GREEN_cons = '\033[32m'
YELLOW_cons = '\033[33m'
BLUE_cons = '\033[34m'
MAGENTA_cons = '\033[35m'
CYAN_cons = '\033[36m'
WHITE_cons = '\033[37m'
UNDERLINE_cons = '\033[4m'
RESET_cons = '\033[0m'
