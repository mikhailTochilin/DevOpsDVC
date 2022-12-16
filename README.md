# DevOps DVC Homework


Имплементация NN модуля _enhancer_toch_, очищающего аудио от посторонних шумов.

В файле _dvc.yaml_ прописаны основные стейджи технологического стека.

1. Установка необходимых библиотек в виртуальное окружение _python==3.8_:
    ```bash
    pip install -r requirements_cuda.txt
    pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    ```
    (последнее требуется для работы с cuda)
2. ```
   dvc repro -s download_data
   ```
   загружает сырые данные для обучения в папку _raw_data_.
3. ```
   dvc repro -s preprocess_data
   ```
   делает простейшую предобработку данных - разархивация (в _valentini_) и создание _json_ файлов (_valentini/egs_) с путями до аудио для подгрузки DataLoader.
4. ```
   dvc repro -s train
   ```
   начинает обучение с параметрами, указанными в файле _conf/config.yaml_ (датасет, # эпох, _batchsize_ и др.) и сохраняет чекпоинты и _hydra logs_ в _outputs/_.
5. ```
   dvc repro -s enhance
   ```
   очищает зашумленные файлы обученной моделью (файлы указаны в _config.yaml_).
6. ```
   dvc repro -s evaluate
   ```
   считает метрику _stoi_ на тестовом датасете обученной моделью и сохраняет в _metrics.json_ (тестовый датасет указан в _config.yaml_). Можно посмотреть улучшение с помощью:
   ```
   dvc metrics show
   dvc metrics diff
   ```

Общий пайплайн:
```
dvc dag:

       +---------------+
       | download_data |
       +---------------+
                *
                *
                *
      +-----------------+
      | preprocess_data |
      +-----------------+
                *
                *
                *
           +-------+
           | train |
           +-------+
          **        **
        **            **
       *                *
+---------+         +----------+
| enhance |         | evaluate |
+---------+         +----------+
```