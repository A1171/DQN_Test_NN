from TraidingRL_Env import TraidingEnvMy,Prioritized_Experience_Replay,Priority_Tree,dueling_DQN_LSTM,LoadFromMultishareToDF,ModelTrainer
import os

#Настройки обработки котировок
SequencyLen=32
Derivative_d=0.6
Derivative_WLen=150
ReduseLoss=0.5
MakePCA=False
QuatePipelineSettings=[SequencyLen,Derivative_d,Derivative_WLen,ReduseLoss,MakePCA]

#Формы данных для подачи в нейросеть. Временная последовательность и текущее направление сделки.
state_shape1 = (SequencyLen, 4) 
state_shape2 = (1) 

#Загрузка датасета
DFTrain,DFTest=LoadFromMultishareToDF(FileName="C:\\Users\\Aleksandr\\AppData\\Local\\Programs\\Python\\Python311\\SplittenModel\\DataFileQuates_10sh_H1.txt",StartYear=2021,ShareNumber=1,TrainPart=0.9)

#Среда с тренировочным и тестовым датасетами
TraidingEnv = TraidingEnvMy(DFTrain,QuatePipelineSettings)
TestingEnv = TraidingEnvMy(DFTest,QuatePipelineSettings)
TraidingEnv.reset()
TestingEnv.reset()

#Создание моделей для улучшенного Q-learning
learning_rate=1e-4
action_size=len(TraidingEnv.ActionSpace)
main_model = dueling_DQN_LSTM(state_shape1,state_shape2, action_size, learning_rate)   #Создаем основную модель
target_model = dueling_DQN_LSTM(state_shape1,state_shape2, action_size, learning_rate) #Создаем целевую модель

    

#Загрузка моделей для продолжения обучения
load_pretrained = True
if load_pretrained:
    if(os.path.isfile('dueling_qn_main2.h5')):
      main_model.load_weights('dueling_qn_main2.h5')     #Загружаем веса основной модели из файла
      target_model.load_weights('dueling_qn_target2.h5') #Загружаем веса целевой модели из файла
    else:
        load_pretrained=False

#Параметры Q-learning
gamma = 0.8                       #Гамма (параметр для передачи наград между состояниями)
initial_epsilon = 1               #Начальное значение эпсилон (вероятность принять рандомный шаг)
#epsilon = initial_epsilon         #Текущее значение эпсилон (инициализируется как стартовое значение)
final_epsilon = 0.01              #Минимальное значение эпсилон (должно быть выше 0)

epsilon_decay_factor = 0.99999    #Задаем число, на которое будем умножать эпсилон после каждого шага  

observation_steps = 30000         #Количество шагов в игровой среде до обучения сети 
target_model_update_rate = 5000   #Веса целевой модели будут обновляться весами основной модели раз в 5 000 шагов

learning_rate = 1e-4   #Обычно в обучении с подкреплением ставят низкий шаг обучения, например 1e-4
batch_size = 64        #Используем размер пакета в 32
memory_capacity=30000
TrainUpdateRate=10
TrainSettings=[gamma,initial_epsilon,final_epsilon,epsilon_decay_factor,observation_steps,target_model_update_rate,learning_rate,batch_size,memory_capacity,TrainUpdateRate]

#Тренер
Trainer=ModelTrainer(TrainSettings,[main_model,target_model],TraidingEnv, load_pretrained=True)
#Тренировка моделей
Trainer.train_cycle()
