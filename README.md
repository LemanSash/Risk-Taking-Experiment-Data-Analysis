main.py - основной файл проекта с анализом данных
data_loader.py - содержит класс, отвечающий за загрузку данных
analyzers.py - содержит классы, отвечающие за обработку данных опросника и поведенческих методик
utils.py - содержит вспомогательные функции предобработки данных и т.д.
models содержит файлы с построенными моделями:
vse_model.py - модель Value plus Sequential Exploration (Ligneul, 2019) для Iowa Gambling Task
stld_model.py - модель Scaled Target Learning with Decay (Zhou et al., 2021) для Balloon Analogue Risk Task
prospect_model_hot.py - модель на основе теории перспектив и ожидаемой полезности, Model 1 из Wullhorst et al. (2024) для Columbia Card Task hot
prospect_model_cold.py - модель на основе теории перспектив и ожидаемой полезности, адаптация Model 1 из Wullhorst et al. (2024) для Columbia Card Task cold
regression_model.py - модель линейной регрессии для параметров предыдущих моделей и балла опросника
