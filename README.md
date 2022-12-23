# CVResearch_Test

В репозитории представлено сравнение разных моделей машинного обучения и анализ результатов для решения задачи face recognition на примере датасета Labeled Faces in the Wild. Репозиторий разделен на несколько частей:

     1) Папка reports. 
     
Содержит в себе отчет о проделанной работе в формате pdf. Также содержит в себе анкету, где есть ссылки на примеры кода, который я писал. В папке есть анализ статьи Replacing Labeled Real-image Datasets with Auto-generated Contours

     2) Папка notebook.
     
Содержит в себе jupyter-notebook в формате ipynb, который использовался для задания моделей и получения результатов. В ноутбуке есть кратко описание задачи, описание моделей, экспериментов, есть архитектуры нейросети, использованной в работе. Также присутствует сам код, который готов к запуску.

Для запуска кода в notebook следует иметь в виду: а) необходимо иметь установленные библиотеки: numpy, sklearn, torch и другие. б) Необходим доступ к GPU. Все вычисления в работе были проведены на видеокартах, предоставленных Google Collab.

    3) Папка pyfiles. 
    
Содержит в себе реализацию моделей и функций обучения в виде .py файлов. 
