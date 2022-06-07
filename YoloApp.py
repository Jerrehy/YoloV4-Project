import streamlit as st
import tempfile
import cv2 as cv
import datetime
import time

st.title("Приложение для обнаружения и определения объектов на видео")
st.markdown("Используется детектор YOLOv4")

st.sidebar.title("Параметры запуска")
st.sidebar.markdown("Выбор параметров запуска")

uploaded_file = st.file_uploader(label="Upload Video", type=["mp4", "avi"])

# Выбор вычислительных средств
net_choice = st.sidebar.radio('Вычислительные средства', ('CPU', 'GPU'))

# Пороги нахождения объекта - вероятность определения
Conf_threshold = st.sidebar.slider('Доверительный порог, %', 0, 100, 40) / 100
NMS_threshold = st.sidebar.slider('Порог Non Maximux Suppression, %', 0, 100, 40) / 100

# Цвета для различных объектов
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Список всех объектов нейронной сети
# Которые нейронная сеть может определять
class_name = []
with open('dnn_model/classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Считывание нейронной сети - файл весов и файл конфигурации
net = cv.dnn.readNet('dnn_model/yolov4.weights', 'dnn_model/yolov4.cfg')

# Основа обработки - база
if net_choice == 'GPU':
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

elif net_choice == 'CPU':
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Создание модели рабочей нейронной сети
model = cv.dnn_DetectionModel(net)

# Параметры работы модели нейройнной сети
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

process_button = st.button('Начать обработку')

stframe = st.empty()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    vf = cv.VideoCapture(tfile.name)

    # Подбор разметки для изображения
    frame_width = int(vf.get(3))
    frame_height = int(vf.get(4))

    size = (frame_width, frame_height)

    if process_button and vf:
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        result = cv.VideoWriter(f'detections/detect-result-{now}.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, size)

        # Просмотр FPS в секунду
        starting_time = time.time()
        frame_counter = 0

        while True:
            # Считывание по кадрам
            ret, frame = vf.read()
            frame_counter += 1

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            counts = dict()
            total_count = 0

            if ret == False:
                break

            classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

            # Подсчёт количества объектов
            for i in classes:
                if i in [2, 3, 5, 7]:
                    obj_class = class_name[i[0]]
                    counts[obj_class] = counts.get(obj_class, 0) + 1
                    total_count += 1
                else:
                    continue

            for (classid, score, box) in zip(classes, scores, boxes):

                color = COLORS[int(classid) % len(COLORS)]

                if classid[0] in [2, 3, 5, 7]:
                    label = "%s : %f" % (class_name[classid[0]], score)
                    cv.rectangle(frame, box, color, 1)
                    cv.putText(frame, label, (box[0], box[1] - 10),
                               cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
                else:
                    continue

            endingTime = time.time() - starting_time
            fps = frame_counter / endingTime

            cv.putText(frame, f'FPS: {fps}', (20, 50),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            cv.putText(frame, f'total count: {total_count}', (20, 70),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            y = 90

            for key in counts:
                cv.putText(frame, f'count {key}: {counts[key]}', (20, y),
                           cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                y += 20

            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            result.write(frame)

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            stframe.image(frame)
