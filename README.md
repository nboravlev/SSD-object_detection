# SSD-object_detection
Детекция и классификация объектов. Архитектура SSD (Single Shot MultiBox Detector)

[Статья](https://d2l.ai/chapter_computer-vision/ssd.html) на тему SSD

Данные для обучения модели содержаться в репозитории BSSD_dataset 
```bash
!git clone https://github.com/Shenggan/BCCD_Dataset.git
```
Датасет представляет из себя изображения клеток крови под микроскопом с разметкой.
![клеток крови под микроскопом с разметкой.](https://github.com/nboravlev/SSD-object_detection/blob/main/example.jpg)

Описание датасета и процесс предобработки данных также представлен в указанном репозитории

Для реализации необходимо выполнить последовательно скрипты:

* utils.py - загружает все необходимые функции
* create_date_lists.py - готовит список данных для обучения и проверки
* datesets.py - формирует датасет для обучения и проверки
* model.py - формирует модель для обучения (VGG16 c дополнительными свертками)
* train.py - обучение модели
* detect.py - функция для формирования предсказаний
* eval.py - проверка модели на валидации
* control.py - контрольный вывод изображения с изначальной разметкой и предсказанной разметкой и классами для контроля

Предобученную на 110 эпохах модель можно скачать по [ссылке](https://drive.google.com/file/d/1-8ZYsqaUzNFxlYtQKGxYTHpx6tcRtVLs/view)
