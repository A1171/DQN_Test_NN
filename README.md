# DQN_Test_NN
DQN share traiding. First test

Задача: Сформулируйте систему наград для алгоритма такую, что агент будет награждаться при открытии позиций, при HOLD и при закрытии с учётом типа сделок.
Напишите на Python фрагмент кода, реализующий предложенную вами систему наград.
Укажите, какие метрики будете использовать для оценки эффективности предложенной системы наград.

Ожидаемый результат:
Документ, описывающий концепцию системы наград и обоснование подхода.
Код на Python, демонстрирующий расчет награды.


Т.к. на каждой свече агент имеет возможность открыть/закрыть сделку или перевернуться, наградой можно выбрать изменение баланса через 1 свечу после действия.
Другой тип наград делать не вижу смысла, т.к. это не тот случай, где множество ранее принятых действий приводит к отдаленному результату.
Здесь результат принятого действия будет ясен моментально, через 1 свечу, при шаге равном 1 свече.

Награда:
1. В случае отсутствия позиции и выбора HOLD Reward=0
2. В случае наличия позиции и выбора HOLD Reward=(Close[i]-Close[i-1])*Direction
Direction=1(Position=Long) Direction=-1(Position=Short)
3. В случае открытия позиции если отсутствовала Reward=(Close[i]-Close[i-1]) * Direction - Comission * Close[i-1]
Comission - комиссия в долях от объема сделки. В коде 0.001(0.1%)
4. В случае переворота позиции Reward=(Close[i]-Close[i-1])*Direction - 2*Comission*Close[i-1]
Комиссия удваивается т.к. проторгован двойной объем.

Код функции смотрите строка 75 файла TraidingRL_Env.py. Функция def Step(self): класса TraidingEnvMy. Функция получает на вход действие (BUY, HOLD, SELL) (1,0,-1)
делает шаг по времени на 1 свечу вперед и считает награду. Разделять функцию отдельно на шаг и награду не стал, т.к. код почти удваивается. там 40 строк не сложно разобраться.

Вместо метрик буду использовать результат торговли, т.к. это обучение с подкреплением торговля постоянно идет, при полном проходе датасета легко посчитать 
результат торговли: прибыль и профит фактор. Еще посмотрю усредненный лосс сети за эпизод на тренировочной и тестовой части датасета. Здесь не сделано.

Ниже скопирована статистика тренировки, видно что во время тренировки прибыль уходит из отрицательной зоны. Профит фактор тоже растет.
Тест проводил на 5 минутке акции газпрома с ян2021г.
Но вообще надо сделать оценку прибыли и профит фактора на тестовой части датасета, чтобы видеть что не переучиваемся, здесь не сделано и скорее всего идет переобучение, но про работу с переобучением в задаче ни чего не сказано.

В файлt A2C_My.py можете посмотреть создание классов и запуск тренировки. Файл маленький 62 строки с комментариями.

Закончился 1-й эпизод. Значение эпсилон: 1, SummReward: -1914.0241699999806, PF: 0.6442046717006001
Закончился 2-й эпизод. Значение эпсилон: 1, SummReward: -2036.3002899999683, PF: 0.6268616726172127
WARNING:tensorflow:From C:\Users\Aleksandr\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

Закончился 3-й эпизод. Значение эпсилон: 0.95, SummReward: -1954.6756099999952, PF: 0.6375653412881856
Закончился 4-й эпизод. Значение эпсилон: 0.85, SummReward: -1784.4415899999958, PF: 0.6618006594054944
Закончился 5-й эпизод. Значение эпсилон: 0.76, SummReward: -1569.8020499999711, PF: 0.6944516531480246
Закончился 6-й эпизод. Значение эпсилон: 0.67, SummReward: -1281.5316699999648, PF: 0.7417672314704826
Закончился 7-й эпизод. Значение эпсилон: 0.6, SummReward: -1289.270009999981, PF: 0.7395278375750058
Закончился 8-й эпизод. Значение эпсилон: 0.53, SummReward: -1008.7807699999803, PF: 0.7881544024071154
Закончился 9-й эпизод. Значение эпсилон: 0.48, SummReward: -503.6832299999637, PF: 0.8880206677910806
Статистика успешно сохранена.
Закончился 10-й эпизод. Значение эпсилон: 0.42, SummReward: -402.88023999995767, PF: 0.909008040935309
Закончился 11-й эпизод. Значение эпсилон: 0.38, SummReward: -546.3562499999748, PF: 0.8783540863615646
Закончился 12-й эпизод. Значение эпсилон: 0.34, SummReward: -367.6247399999843, PF: 0.9162090181009244
Закончился 13-й эпизод. Значение эпсилон: 0.3, SummReward: -292.7356299999783, PF: 0.9324713886605843
Закончился 14-й эпизод. Значение эпсилон: 0.27, SummReward: -114.01676999997562, PF: 0.9730175848212237
Закончился 15-й эпизод. Значение эпсилон: 0.24, SummReward: -339.3193699999806, PF: 0.9217472832769309
Закончился 16-й эпизод. Значение эпсилон: 0.21, SummReward: -31.108069999968393, PF: 0.9925153935197634
Закончился 17-й эпизод. Значение эпсилон: 0.19, SummReward: 104.98459000003277, PF: 1.0256514058673432
Закончился 18-й эпизод. Значение эпсилон: 0.17, SummReward: 52.6390500000125, PF: 1.0128101333215112
Закончился 19-й эпизод. Значение эпсилон: 0.15, SummReward: 43.01669000002039, PF: 1.0104897968646258
Статистика успешно сохранена.
Закончился 20-й эпизод. Значение эпсилон: 0.13, SummReward: 227.54664000003413, PF: 1.0569348747293243
Закончился 21-й эпизод. Значение эпсилон: 0.12, SummReward: 18.917650000030335, PF: 1.0046097280448765
Закончился 22-й эпизод. Значение эпсилон: 0.11, SummReward: 257.3099500000217, PF: 1.0647571031208485
Закончился 23-й эпизод. Значение эпсилон: 0.09, SummReward: 352.72431000003553, PF: 1.089792963788473
Закончился 24-й эпизод. Значение эпсилон: 0.08, SummReward: 235.32319000002644, PF: 1.059051455220357
Закончился 25-й эпизод. Значение эпсилон: 0.07, SummReward: 445.6909300000316, PF: 1.1149634009002916
Закончился 26-й эпизод. Значение эпсилон: 0.07, SummReward: 417.42749000002823, PF: 1.1075302246486458
Закончился 27-й эпизод. Значение эпсилон: 0.06, SummReward: 564.5278100000332, PF: 1.14845553380031
Закончился 28-й эпизод. Значение эпсилон: 0.05, SummReward: 486.7456100000345, PF: 1.1268768497322579
Закончился 29-й эпизод. Значение эпсилон: 0.05, SummReward: 588.2460700000347, PF: 1.155035520265253
Статистика успешно сохранена.
Закончился 30-й эпизод. Значение эпсилон: 0.04, SummReward: 636.3691300000355, PF: 1.1690814546351764
Закончился 31-й эпизод. Значение эпсилон: 0.04, SummReward: 675.8852700000352, PF: 1.1808861473895806
Закончился 32-й эпизод. Значение эпсилон: 0.03, SummReward: 631.0531700000306, PF: 1.1676287841144686
Закончился 33-й эпизод. Значение эпсилон: 0.03, SummReward: 631.6354500000407, PF: 1.167785679350868
Закончился 34-й эпизод. Значение эпсилон: 0.03, SummReward: 709.3177100000385, PF: 1.1902921757033105
Закончился 35-й эпизод. Значение эпсилон: 0.02, SummReward: 599.6054700000277, PF: 1.1587584519206429
Закончился 36-й эпизод. Значение эпсилон: 0.02, SummReward: 690.5639700000329, PF: 1.1850091982847941
Закончился 37-й эпизод. Значение эпсилон: 0.02, SummReward: 754.271330000045, PF: 1.2039448803126551
Закончился 38-й эпизод. Значение эпсилон: 0.02, SummReward: 810.0698300000477, PF: 1.220919267554323
Закончился 39-й эпизод. Значение эпсилон: 0.01, SummReward: 758.0048100000413, PF: 1.2052780975328061
Статистика успешно сохранена.
Закончился 40-й эпизод. Значение эпсилон: 0.01, SummReward: 747.5796700000355, PF: 1.2023226853722435
Закончился 41-й эпизод. Значение эпсилон: 0.01, SummReward: 873.413370000043, PF: 1.2403803421528792
Закончился 42-й эпизод. Значение эпсилон: 0.01, SummReward: 769.567550000047, PF: 1.2087351975150846
Закончился 43-й эпизод. Значение эпсилон: 0.01, SummReward: 750.2417300000452, PF: 1.2025361252503912
Закончился 44-й эпизод. Значение эпсилон: 0.01, SummReward: 834.9339300000465, PF: 1.2289283568661777

