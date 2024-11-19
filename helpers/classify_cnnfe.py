"""
Модуль классификатора на базе предобученной
сверточной сети как экстрактора фич CNNfe
"""
import numpy as np
import cv2 as cv
# from PIL import Image, ImageDraw, ImageFont
import os
import datetime
import warnings
import matplotlib.pyplot as plt
#
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
#
import settings as s
import helpers.utils as u
import helpers.dnn as z


###########################################################
# Функции работы с моделью CNN features extractor
###########################################################
def create_embeddings(model, data_folders, preprocess=False, preprocess_func=None):
    """
    Пройти по папкам data_folders и создать там эмбеддинги
    :param model:
    :param data_folders:
    :param preprocess:
    :param preprocess_func: функция препроцессинга
    :return:
    """
    # Если не получили функцию препроцессинга
    if preprocess_func is None:
        preprocess = False

    for folder in data_folders:
        emb_list = []
        for f in sorted(os.listdir(folder)):
            _, file_extension = os.path.splitext(f)
            if os.path.isfile(os.path.join(folder, f)) and file_extension in s.ALLOWED_IMAGES:
                img = cv.imread(os.path.join(folder, f))
                if type(img) is None:
                    print("Не удалось загрузить файл {}".format(os.path.join(folder, f)))
                    continue
                print("Обрабатываем файл: {}".format(os.path.join(folder, f)))
                #
                if preprocess:
                    print("  Используется функция предобработки изображения: {}".format(preprocess_func.__name__))
                    img = preprocess_func(img)
                # u.show_image_cv(img)

                blob = z.get_blob_dnn(img, input_shape=(224, 224), to_gray=False, subtract_mean=False)
                blob = np.transpose(blob, (0, 2, 3, 1))  # (1, 3, 224, 224) -> (1, 224, 224, 3)

                emb = z.get_pred_dnn(model, blob)
                emb_list.append(emb[0])
        # Сохраним массив в соответствующей папке
        emb_np = np.array(emb_list)
        print("Сохраняем массив эмбеддингов в папкe {}".format(folder))
        np.save(os.path.join(folder, 'embeddings'), emb_np)


def collect_embeddings(data_folders):
    """
    Собираем эмбеддинги из папок в один массив numpy
    :return: массив эмбеддингов, указатели классов
    """
    class_pointers = []
    all_embeddings = np.array([])
    print("  Загружаем эмбеддинги в папке:")
    for idx, folder in enumerate(data_folders):
        print("  {}".format(folder))
        curr_emb = np.load(os.path.join(folder, 'embeddings.npy'))
        #
        if idx == 0:
            all_embeddings = curr_emb
            class_pointers.append(curr_emb.shape[0])
        else:
            all_embeddings = np.concatenate([all_embeddings, curr_emb], axis=0)
            class_pointers.append(class_pointers[-1] + curr_emb.shape[0])
    # print("Собрали all_embeddings.shape:", all_embeddings.shape)
    # print("class_pointers:", class_pointers)
    return all_embeddings, class_pointers


def create_tsne_scatter(labels, all_embeddings, class_pointers):
    """
    Создать скаттер
    :param labels:
    :param all_embeddings:
    :param class_pointers:
    :return:
    """
    # Фильтруем warnings от TSNE
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        tsne = TSNE(n_components=2).fit_transform(all_embeddings)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = u.scale_to_01_range(tx)
    ty = u.scale_to_01_range(ty)
    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Сформируем палитру по числу классов
    cmap = u.get_cmap(len(labels) + 1)  # +1!
    # Рисуем точки
    start_class_idx = 0
    for idx, label in enumerate(labels):
        # print(idx, label, start_class_idx, class_pointers[idx])
        curr_tx = tx[start_class_idx: class_pointers[idx]]
        curr_ty = ty[start_class_idx: class_pointers[idx]]
        #
        start_class_idx = class_pointers[idx]
        #
        color = cmap(idx)
        color = np.array(color).reshape(1, -1)  # prevent warning from plt
        ax.scatter(curr_tx, curr_ty, c=color, label=labels[idx])

    # Сохраняем scatter
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    try:
        plt.savefig('scatter1.png', dpi=600)
        print("Сохранили scatter1.png")
    except:
        print("Ошибка записи scatter1.png")


def cnnfe_preparation(emb_path, folders_to_include=None):
    """
     Подготовка к запуску CNN детектора форм.
     Пройти по папкам датасета и собрать эмбеддинги в один массив
    :param folders_to_include:
    :param emb_path:
    :param folders_to_include: использовать только папки из списка
    :return:
    """
    # Если ограниченный список не задан или пуст, то обходим все папки с классами
    if folders_to_include is None or len(folders_to_include) == 0:
        data_folders = []  # полные пути к папкам
        labels = []  # имена папок как метки классов
        for f in sorted(os.listdir(emb_path)):
            if os.path.isdir(os.path.join(emb_path, f)):
                # print("Подключаем класс в папке: {}".format(f))
                data_folders.append(os.path.join(emb_path, f))
                labels.append(f)
    else:
        # Иначе обходим папки из ограниченного списка
        data_folders = []  # полные пути к папкам
        labels = []  # имена папок как метки классов
        for f in sorted(os.listdir(emb_path)):
            if os.path.isdir(os.path.join(emb_path, f)) and (f in folders_to_include):
                # print("Подключаем класс в папке: {}".format(f))
                data_folders.append(os.path.join(emb_path, f))
                labels.append(f)

    print("Нашли папок классов: {}".format(len(data_folders)))
    # Пройдем по сохраненным эмбеддингам и соберем их в один массив
    all_embeddings, class_pointers = collect_embeddings(data_folders)
    return data_folders, labels, all_embeddings, class_pointers


def get_class_label(pred_idx, class_pointers, labels):
    """
    Определяет класс (label) по индексу
    :param pred_idx: индекс
    :param class_pointers: список индексов, на которых сменяются классы в all_embeddings
    :param labels: список меток классов
    :return: метка класса
    """
    for idx in range(len(class_pointers)):
        if class_pointers[idx] > pred_idx:
            # labels[idx] == os.path.basename(data_folders[idx])
            return labels[idx]


def cnnfe_classifier(model,
                     n_votes_fe,
                     labels,
                     all_embeddings,
                     class_pointers,
                     img,
                     put_txt=False,
                     tsne=False,
                     verbose=False):
    """
    Функция классифицирует изображения через эмбеддинги
    :param model: модель экстрактора фич
    :param n_votes_fe: количество соседних точек для голосования
    :param labels: метки классов
    :param all_embeddings: массив эмбеддингов
    :param class_pointers:
    :param put_txt: написать на изображении результат предикта
    :param img: изображение
    :param tsne: делать tsne визуализацию предикта
    :param verbose:
    :return: предикт класса, confidence, размеченное изображение (CV), визуализация TSNE (PIL)
    """
    img_parsed = img.copy()

    blob = z.get_blob_dnn(img, input_shape=(224, 224), to_gray=False, subtract_mean=False)
    blob = np.transpose(blob, (0, 2, 3, 1))  # (1, 3, 224, 224) -> (1, 224, 224, 3)
    pred = z.get_pred_dnn(model, blob)
    cosPred = cosine_similarity(pred, all_embeddings)
    # print(np.argmax(cosPred), cosPred[:, np.argmax(cosPred)])

    # Определим к какому классу принадлежит полученный индекс
    # pred_idx = np.argmax(cosPred)
    # pred_label = get_class_label(pred_idx, class_pointers, labels)
    # print("Классифицирован по БЛИЖАЙШЕЙ точке как: {}".format(pred_label))

    # Определим к какому классу принадлежат N ближайших соседей
    N = n_votes_fe
    indices = np.argsort(cosPred[0])
    class_dict = {}
    for idx in range(1, N + 1):
        curr_idx = indices[-idx]
        curr_label = get_class_label(curr_idx, class_pointers, labels)
        if curr_label in class_dict.keys():
            class_dict[curr_label] += 1
        else:
            class_dict[curr_label] = 1
    # Предикт "по соседям"
    pred_final = max(class_dict, key=class_dict.get)
    pred_conf = round(class_dict[pred_final] / N, 2)
    if verbose:
        print("  {} определен с conf={:.0%} по {} точкам".format(pred_final, pred_conf, N))

    # Напишем название класса и результат голосования
    if put_txt:
        #
        cv.putText(img_parsed,
                   "Class: {}, conf={:.0%} by {} voters".format(pred_final, pred_conf, N),
                   (10, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv.LINE_AA)

    if not tsne:
        return pred_final, pred_conf, img_parsed, None
    else:
        # ##### Добавим к эмбеддингам результат текущего документа
        all_embeddings_pred = np.concatenate([all_embeddings, pred], axis=0)
        # Добавляем указатель класса
        class_pointers.append(class_pointers[-1] + pred.shape[0])
        # Добавляем метку класса
        labels.append('NEAR:' + pred_final)

        # filter warnings from TSNE
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tsne = TSNE(n_components=2).fit_transform(all_embeddings_pred)
        # extract x and y coordinates representing the positions of the images on T-SNE plot
        tx = tsne[:, 0]
        ty = tsne[:, 1]
        tx = u.scale_to_01_range(tx)
        ty = u.scale_to_01_range(ty)
        # initialize a matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Сформируем палитру по числу классов + 1
        cmap = u.get_cmap(len(labels) + 1)
        # Рисуем точки
        start_class_idx = 0
        for idx, label in enumerate(labels):
            # print(idx, label, start_class_idx, class_pointers[idx])
            curr_tx = tx[start_class_idx: class_pointers[idx]]
            curr_ty = ty[start_class_idx: class_pointers[idx]]
            #
            start_class_idx = class_pointers[idx]
            #
            if idx < len(labels) - 1:
                color = cmap(idx)
                color = np.array(color).reshape(1, -1)  # prevent warning from plt
            else:
                color = 'black'
            ax.scatter(curr_tx, curr_ty, c=color, label=labels[idx])

        # Формируем скаттер
        ax.set_title("Класс {} по {} точкам. Голосов:{:.0%}".format(pred_final, N, pred_conf))
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        fig = plt.gcf()
        img_scatter = u.fig2img_cv(fig)

        # Откатим данные текущего предикта
        class_pointers.pop()
        labels.pop()
        #
        return pred_final, pred_conf, img_parsed, img_scatter


# ################# КЛАСС Classifier ######################
class Classifier:
    """
    Класс - обертка для работы с классификатором на основе CNNfe
    """
    # Модель одна для всех экземпляров классификатора
    model_fe = None

    def __init__(self,
                 model,
                 emb_path,
                 preprocess_func=None,
                 conf_threshhold=None,
                 force_data_collect=False,
                 data_collect_path=None,
                 data_collect_limit=0,
                 verbose=False):
        """
        Функция инициализации экземпляра классификатора

        :param model: модель CNNfe
        :param emb_path: путь к данным/эмбеддингам
        :param preprocess_func: функция препроцессинга изображения
        :param conf_threshhold: порог конфиденса
        :param force_data_collect: сохранять результаты предиктов
        :param data_collect_path: путь для сохранения результатов предиктов
        :param data_collect_limit: ограничитель количества сохраненных результатов предиктов (0 - не ограничивать)
        :param verbose:
        """
        if Classifier.model_fe is None:
            Classifier.model_fe = model
        assert Classifier.model_fe is not None, "Нет модели CNNfe"

        # Если путь к данным/эмбеддингам не задан, то работа классификатора невозможна
        assert os.path.isdir(emb_path), "Не найден датасет {}".format(emb_path)
        self.emb_path = emb_path

        # Если не получили функцию препроцессинга
        if preprocess_func is None:
            self.preprocess = False
        else:
            self.preprocess = True
            self.preprocess_func = preprocess_func

        # Если параметр конфиденс не задан, то берем его из настроек по умолчанию
        if conf_threshhold is None:
            self.conf_threshhold = s.CONFIDENCE_THRESHOLD_votes
        else:
            self.conf_threshhold = conf_threshhold

        # Если параметр сохранения данных не задан, то берем его из настроек по умолчанию
        if force_data_collect is None:
            self.force_data_collect = s.CNNFE_DATA_COLLECT
        else:
            self.force_data_collect = force_data_collect

        # Если не существует путь для сохранения данных предиктов, то отменяем сохранение
        if force_data_collect:
            if data_collect_path is None:
                print("Не задан путь для сохранения результатов предиктов, "
                      "сохранение отменено".format(data_collect_path))
                self.force_data_collect = False
            else:
                self.data_collect_path = data_collect_path
        #
        self.data_collect_limit = data_collect_limit

        #
        self.verbose = verbose
        if self.verbose:
            print("Создан CNNfe классификатор в датасете: {}, conf_threshhold={}".format(self.emb_path,
                                                                                         self.conf_threshhold))

        # Загрузим папки, метки классов, массив эмбеддигов, указатели классов
        class_folders_list = []  # если список пуст, то будут загружены все папки с данными
        self.data_folders, self.labels, self.all_embeddings, self.class_pointers = cnnfe_preparation(self.emb_path,
                                                                                                     class_folders_list)
        #
        self.img, self.img_orig, self.pred_cnnfe, self.conf_cnnfe, self.img_cnnfe = None, None, None, None, None

    def __del__(self):
        print("Удален CNNfe классификатор в датасете: {}".format(self.emb_path))

    def display_classificator_info(self):
        print("CNNfe классификатор в датасете: {}, conf_threshhold={}".format(self.emb_path,
                                                                              self.conf_threshhold))

    def get_pred(self, img):
        """
        Функция получения предикта
        :param img: изображение
        :return: предикт, конфиденс
        """
        self.img_orig = img.copy()

        if self.preprocess:
            self.img = self.preprocess_func(img)
        else:
            self.img = img.copy()
        #
        self.pred_cnnfe, self.conf_cnnfe, self.img_cnnfe, _ = cnnfe_classifier(Classifier.model_fe,
                                                                               self.conf_threshhold,
                                                                               self.labels,
                                                                               self.all_embeddings,
                                                                               self.class_pointers,
                                                                               self.img,
                                                                               put_txt=False,
                                                                               tsne=False,
                                                                               verbose=self.verbose)

        if self.force_data_collect:
            self.save_collected_data()
        return self.pred_cnnfe, self.conf_cnnfe, self.img_cnnfe

    def save_collected_data(self):
        """
        Функция сохранения результата предикта
        """
        if os.path.basename(self.data_collect_path) not in os.listdir(s.EMB_PATH):
            os.mkdir(self.data_collect_path)
            print("Создали папку для сохранения предиктов CNN_fe: {}".format(self.data_collect_path))
        else:
            print("Результаты предиктов CNN_fe сохраняются в папке: {}".format(self.data_collect_path))
        #
        if self.pred_cnnfe not in os.listdir(self.data_collect_path):
            os.mkdir(os.path.join(self.data_collect_path, self.pred_cnnfe))
            print("Создали папку для сохранения предиктов класса: {}".format(self.pred_cnnfe))

        # Подготовим уникальное имя
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")
        target_base_name = "_".join(["collected", suffix, ".png"])
        target_full_name = os.path.join(self.data_collect_path, self.pred_cnnfe, target_base_name)
        #
        if cv.imwrite(target_full_name, self.img_orig):
            print("Сохранили копию файла: {}".format(target_full_name))
        else:
            print("Не удалось сохранить копию файла: {}".format(target_full_name))
        #
        N_limit = self.data_collect_limit
        if N_limit > 0:
            collected_files = os.listdir(os.path.join(self.data_collect_path, self.pred_cnnfe))
            # print(collected_files)
            if len(collected_files) > N_limit:
                N_to_remove = len(collected_files) - N_limit
                N_to_remove = max(N_to_remove, 0)
                if N_to_remove > 0:
                    print("  Лимит количества файлов в папке класса: {},"
                          " к удалению: {}".format(N_limit, N_to_remove))
                    #
                    collected_files = \
                        [os.path.join(self.data_collect_path, self.pred_cnnfe, x) for x in collected_files]
                    collected_files = sorted(collected_files, key=os.path.getmtime)
                    #
                    for file_to_remove in collected_files[:N_to_remove]:
                        print("  Удаляем файл: {}".format(file_to_remove))
                        os.remove(file_to_remove)
