# Object-Detection-Eval

Чтобы воспользоваться скриптом для оценки качества модели, надо просто запустить `python model_detection_eval.py` с аргументами:
- "-h", который показывает help-message
- "-t" и указать в кавычках через пробел значения трешхолда для IOU, которые надо использовать (ex.: '-t "0.5 0.6 0.9"'), по дефолту будет использоваться только 0.5
- "-gt_path", чтобы указать путь к файлу с ground truth 
- "-pred_path", чтобы указать путь к файлу с предиктами модели
- "-output_path", чтобы указать в какой файл записать результаты теста моделей

Пример ввода:
`python model_detection_eval.py -gt_path val_small_40_classes.json -pred_path coco_instances_results.json -output_path output.csv -t "0.5 0.6 0.9"`


В результирующем файлике в первых N строках (где N - количество классов) будут посчитаны различные метрики по классам, такие как: AP, Accuracy, Precision, Recall, вместе с TP, FP и total_P (количество ground truth bbox-ов) для разных порогов IOU. Столбцы в данной таблице будут иметь название вида "{MetricName}@{IOU_T\*100}".
Потом через пустую строчку будут метрики mAP и Accuracy, усредненные по классам для разных порогов IOU. В самом низу таблицы будут метрики mAP и AR, усредненные по IOU=[.5 .. .95]. Пример выхода лежит в файлике `output.csv`.

**Важно**: название колонок в таблице напрямую относится к тем строкам, где указан класс. Если в ячейке класса указана метрика, то справа от нее будет значение (одно или несколько) данной метрики.

Пример части таблички:
![Image of Table Part](https://ia.wampi.ru/2020/09/29/Screen-Shot-2020-09-29-at-17.50.21.png)
