<h1>Анализ параметров когнитивных моделей принятия рискованных решений</h1>
<h4>Участники прошли опросник на импульсивность Барратта и 4 поведенческие методики на рискованное поведение: Balloon Analohue Risk Task, Iowa Gambling Task, Columbia Card Task cold & Columbia Card Task hot.</h4>
<h5>Был проведён корреляционный анализ классических методов оценки производительности в поведенческих методиках с баллами импульсивности и друг с другом. Далее были построены математические модели под каждую методику, для каждого участника были подобраны параметры данных моделей. Был проведён факторный и корреляционный анализ параметров моделей.</h5>
<b>main.py</b> - основной файл проекта с анализом данных<br>
<b>data_loader.py</b> - содержит класс, отвечающий за загрузку данных<br>
<b>analyzers.py</b> - содержит классы, отвечающие за обработку данных опросника и поведенческих методик<br>
<b>utils.py</b> - содержит вспомогательные функции предобработки данных и т.д.<br>
<b>models</b> содержит файлы с построенными моделями:<br>
<ul>
  <li><b>ev_model.py</b> - модель Expectancy Valence (EV) Model (Steingroever et al., 2014) для Iowa Gambling Task</li><br>
  <li><b>pvl_model.py</b> - модель Prospect Valence Learning Model (PVL) (Steingroever et al., 2014) для Iowa Gambling Task</li><br>
  <li><b>vpp_model.py</b> - модель Value-Plus-Perseveration (VPP) Model (Steingroever et al., 2016) для Iowa Gambling Task</li><br>
  <li><b>vse_model.py</b> - модель Value plus Sequential Exploration (Ligneul, 2019) для Iowa Gambling Task</li><br>
  <li><b>orl_model.py</b> - модель Outcome-Representation Learning (ORL) Model (Haines et al., 2018) для Iowa Gambling Task</li><br>
  <li><b>wallsten_model.py</b> - модель Wallsten Model 3 (Wallsten et., 2005) для Balloon Analogue Risk Task</li><br>
  <li><b>ew_model.py</b> - модель Exponential Weight (EW) Model (Park et al., 2021) для Balloon Analogue Risk Task</li><br>
  <li><b>ewmv_model.py</b> - модель Exponential Weight Mean Variance (EWMV) Model (Park et al., 2021) для Balloon Analogue Risk Task</li><br>
  <li><b>par4_model.py</b> - модель Par4 (Park et al., 2021) для Balloon Analogue Risk Task</li><br>
  <li><b>stl_model.py</b> - модель Scaled Target Learning (STL) Model (Zhou et al., 2021) для Balloon Analogue Risk Task</li><br>
  <li><b>stld_model.py</b> - модель Scaled Target Learning with Decay (Zhou et al., 2021) для Balloon Analogue Risk Task</li><br>
  <li><b>wullhorst_model1_hot.py</b> - модель Model 1 из Wullhorst et al. (2024) для Columbia Card Task hot</li><br>
  <li><b>wullhorst_model1_cold.py</b> - адаптация Model 1 из Wullhorst et al. (2024) для Columbia Card Task cold</li><br>
  <li><b>wullhorst_model2_hot.py</b> - модель Model 2 из Wullhorst et al. (2024) для Columbia Card Task hot</li><br>
  <li><b>wullhorst_model2_cold.py</b> - адаптация Model 2 из Wullhorst et al. (2024) для Columbia Card Task cold</li><br>
  <li><b>haffke_model_hot.py</b> - модель Haffke Model 3 (Haffke & Hübner, 2019) для Columbia Card Task hot</li><br>
  <li><b>haffke_model_cold.py</b> - адаптация модели Haffke Model 3 (Haffke & Hübner, 2019) для Columbia Card Task cold</li><br>
  <li><b>regression_model.py</b> - модель линейной регрессии для параметров предыдущих моделей и балла опросника</li>
</ul>
