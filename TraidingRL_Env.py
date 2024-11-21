import tensorflow as tf       
import numpy as np            # Библиотека NumPy
import random                 # Импортируем модуль для генерации рандомных значений
import pickle                 # Модуль для сохранения результатов в файл
from collections import deque   # deque — это список, где автоматически удаляются старые значения при добавлении новых, чтобы не было переполнения памяти

import keras
#from tensorflow.keras.models import load_model, Model, Sequential # Импортируем функции для создания и загрузки модели из TensorFlow
#from tensorflow.keras.layers import *                             # Импортируем все слои из Keras
#from tensorflow.keras.optimizers import RMSprop, Adam             # Импортируем оптимизаторы RMSprop и Adam
from tensorflow.keras import backend as K                         # Импортируем модуль для бэкэнда Keras
#from tensorflow.keras.utils import to_categorical                 # Импортируем функцию для удобного onehot-энкодинга
#from moviepy.editor import *                                      # Импортируем полезные функции из библиотеки для воспроизведения видео с результатом
#from google.colab import output                                   # Импортируем функцию для управления вывода в Colab-ячейках

import heapq
import numpy as np
from itertools import count
from datetime import datetime
import pandas as pd
import random
import os

class TraidingEnvMy():
    def TransformToStationary_v2(self,InputArray,d=0.6,WLen=150):
      w=[1]
      for k in range(1,WLen):
        w.append(-w[-1]*(d-k+1)/k)
      w=np.array(w)
      #print(w)
      NewArray=[]
      for i in range(InputArray.shape[1]):
          NewArray.append(np.convolve(InputArray[:,i].reshape(-1),w, 'valid'))
      NewArray=np.array(NewArray).transpose()
      return NewArray
    def GetRecord(self,Pos):
        return self.signal_featuresSt[Pos-self.SequencyLen+1:Pos+1],self.signal_times[Pos]
    def __init__(self,df,QuatePipelineSettings):
        #super().__init__(verbose)
        self.SequencyLen,self.Derivative_d,self.Derivative_WLen,self.ReduseLoss,MakePCA=QuatePipelineSettings
        self.signal_features = df.loc[:, ['Open', 'High', 'Low','Close', 'Volume']].to_numpy()
        self.CloseClose=np.zeros(len(self.signal_features))
        print(self.CloseClose.shape,(self.signal_features[1:,3]-self.signal_features[:-1,3]).shape)
        self.CloseClose[1:]=self.signal_features[1:,3]-self.signal_features[:-1,3]
        self.MeanPriceChange,self.STDPriceChange=self.CloseClose.mean(),self.CloseClose.std()
        self.CloseCloseNorm=np.zeros(len(self.signal_features))
        self.CloseCloseNorm=self.CloseClose/self.STDPriceChange
        self.signal_times = df.index.to_numpy()#df.loc[:, ['Date']].to_numpy()
        self.signal_featuresSt = self.TransformToStationary_v2(self.signal_features[:,:4],d=self.Derivative_d,WLen=self.Derivative_WLen)

        self.ActionSpace=[-1,0,1]
        self.PositionPriceOpen=0
        self.Comission=0.001
        self.Comission=self.Comission/self.STDPriceChange
        _=self.reset()
        
    def reset(self):
        self.RecordPtr=self.SequencyLen-1
        self.SummProfits=0
        self.SummLosses=0
        self.CurrentPosition=0
        self.LastReward=0
        self.PositionPriceOpen=0
        return self.GetRecord(self.RecordPtr),0,False
    def get_state(self):
        envState,envTime=self.GetRecord(self.RecordPtr)
        return [envState,self.CurrentPosition]
    #@property 
    def get_variables(self):
        PF=0 if self.SummLosses<=0 else self.SummProfits/self.SummLosses
        Profit=self.SummProfits-self.SummLosses
        return [Profit,PF]
    def get_last_reward(self):
        return self.__LastReward
    def Step(self,Action):
        Reward=0
        self.LastAction=Action
        if(Action==0):#Hold
            self.RecordPtr+=1
            Reward=self.CloseCloseNorm[self.RecordPtr]*self.CurrentPosition
        elif(Action==1):#Buy
            
            if(self.CurrentPosition==0):#Position=0 now, do nothing, Reward=(Close[i]-Close[i-1])*Direction
                self.PositionPriceOpen=self.signal_features[self.RecordPtr,3]
                self.CurrentPosition=1
                self.RecordPtr+=1
                Reward=self.CloseCloseNorm[self.RecordPtr]*self.CurrentPosition-self.Comission*self.PositionPriceOpen
            elif(self.CurrentPosition==-1):#Position short now, invert position, Reward=(Close[i]-Close[i-1])*Direction-Comission*self.Close[i-1]*2
                self.PositionPriceOpen=self.signal_features[self.RecordPtr,3]
                self.CurrentPosition=1
                self.RecordPtr+=1
                Reward=self.CloseCloseNorm[self.RecordPtr]*self.CurrentPosition-self.Comission*self.PositionPriceOpen*2
            elif(self.CurrentPosition==1):#Position long now, invert position, Reward=(Close[i]-Close[i-1])*Direction
                self.RecordPtr+=1
                Reward=self.CloseCloseNorm[self.RecordPtr]*self.CurrentPosition
        elif(Action==-1):#Sell
            if(self.CurrentPosition==0):#Position=0 now, do nothing, Reward=(Close[i]-Close[i-1])*Direction
                self.PositionPriceOpen=self.signal_features[self.RecordPtr,3]
                self.CurrentPosition=-1
                self.RecordPtr+=1
                Reward=-self.CloseCloseNorm[self.RecordPtr]*self.CurrentPosition-self.Comission*self.PositionPriceOpen
            elif(self.CurrentPosition==1):#Position short now, invert position, Reward=(Close[i]-Close[i-1])*Direction-Comission*self.Close[i-1]*2
                self.PositionPriceOpen=self.signal_features[self.RecordPtr,3]
                self.CurrentPosition=1
                self.RecordPtr+=1
                Reward=-self.CloseCloseNorm[self.RecordPtr]*self.CurrentPosition-self.Comission*self.PositionPriceOpen*2
            elif(self.CurrentPosition==-1):#Position short now, invert position, Reward=(Close[i]-Close[i-1])*Direction
                self.RecordPtr+=1
                Reward=-self.CloseCloseNorm[self.RecordPtr]*self.CurrentPosition
        self.Done=self.RecordPtr>=len(self.signal_featuresSt)-1
        envState,envTime=self.GetRecord(self.RecordPtr)
        if(Reward>0):
            self.SummProfits+=Reward*self.STDPriceChange
        else:
            self.SummLosses-=Reward*self.STDPriceChange
        self.__LastReward=Reward
        return [envState,self.CurrentPosition],Reward,self.Done
        
def dueling_DQN_CONVLSTM(input_shape1,input_shape2, action_size, learning_rate):
  state_input1 = keras.layers.Input(shape=(input_shape1))
  state_input2 = keras.layers.Input(shape=(input_shape2))
  x = keras.layers.Conv1D(32, 3,padding="valid", activation='tanh',)(state_input1)
  x = keras.layers.Conv1D(32, 3,padding="valid", activation='tanh',)(x)
  x = keras.layers.MaxPooling1D(2)(x)
  x = keras.layers.LSTM(32,return_sequences=False)(x)
  x = keras.layers.Concatenate()([x,state_input2])

  # Ветка значения состояния — пытается предсказать значения состояния V(s)
  state_value = keras.layers.Dense(256, activation='relu')(x)          #Добавляем скрытый полносвязный слой с 256 нейронами
  state_value = keras.layers.Dense(1)(state_value)                     #Добавляем полносвязный слой с одним нейроном, который будет считать скалярное значение V(s)
  #Нам нужно добавить размерность к этому слою для дальнейшего суммирования с веткой преимущества. Это делается через лямбда-слой
  state_value = keras.layers.Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_size,))(state_value) 

  # Ветка преимущества действия — пытается предсказать значения преимущества А(s, а) для каждого возможного действия а
  action_advantage = keras.layers.Dense(256, activation='relu')(x)     #Добавляем скрытый полносвязный слой с 256 нейронами
  action_advantage = keras.layers.Dense(action_size)(action_advantage) #Добавляем полносвязный слой с action_size количеством нейронов (action_size — это количество уникальных возможных действий)
  #Чтобы заставить эту ветку считать преимущества, мы добавляем самописную функцию, которая вычитывает среднее значение. Таким образом, все преимущества 
  #которые ниже среднего становятся отрицательными, а все значения выше среднего остаются положительными.
  action_advantage = keras.layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(action_advantage) 

  # Суммируем преимущества и значения состояний, чтобы получить Q(s, a) для каждого возможного действия а
  state_action_value = keras.layers.Add()([state_value, action_advantage])
  optimizer = keras.optimizers.legacy.RMSprop(learning_rate = learning_rate) 
  #optimizer = keras.optimizers.legacy.Adam(learning_rate = learning_rate)

  loss_function = tf.keras.losses.Huber(delta = 2)
  model = keras.models.Model([state_input1,state_input2], state_action_value) #Создаем модель, которая принимает на вход состояние среды и возвращает все значения Q(s, a)
  model.compile(loss = loss_function, optimizer = optimizer) #Компилируем модель, используя функцию ошибки, которую объявляем ниже
  return model   #Функция возвращает модель
def dueling_DQN_LSTM(input_shape1,input_shape2, action_size, learning_rate):
  state_input1 = keras.layers.Input(shape=(input_shape1))
  state_input2 = keras.layers.Input(shape=(input_shape2))
  x = keras.layers.LSTM(32,return_sequences=False)(state_input1)
  x = keras.layers.Concatenate()([x,state_input2])

  # Ветка значения состояния — пытается предсказать значения состояния V(s)
  state_value = keras.layers.Dense(256, activation='relu')(x)          #Добавляем скрытый полносвязный слой с 256 нейронами
  state_value = keras.layers.Dense(1)(state_value)                     #Добавляем полносвязный слой с одним нейроном, который будет считать скалярное значение V(s)
  #Нам нужно добавить размерность к этому слою для дальнейшего суммирования с веткой преимущества. Это делается через лямбда-слой
  state_value = keras.layers.Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_size,))(state_value) 

  # Ветка преимущества действия — пытается предсказать значения преимущества А(s, а) для каждого возможного действия а
  action_advantage = keras.layers.Dense(256, activation='relu')(x)     #Добавляем скрытый полносвязный слой с 256 нейронами
  action_advantage = keras.layers.Dense(action_size)(action_advantage) #Добавляем полносвязный слой с action_size количеством нейронов (action_size — это количество уникальных возможных действий)
  #Чтобы заставить эту ветку считать преимущества, мы добавляем самописную функцию, которая вычитывает среднее значение. Таким образом, все преимущества 
  #которые ниже среднего становятся отрицательными, а все значения выше среднего остаются положительными.
  action_advantage = keras.layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(action_advantage) 

  # Суммируем преимущества и значения состояний, чтобы получить Q(s, a) для каждого возможного действия а
  state_action_value = keras.layers.Add()([state_value, action_advantage])
  optimizer = keras.optimizers.legacy.RMSprop(learning_rate = learning_rate) 
  #optimizer = keras.optimizers.legacy.Adam(learning_rate = learning_rate)

  loss_function = tf.keras.losses.Huber(delta = 2)
  model = keras.models.Model([state_input1,state_input2], state_action_value) #Создаем модель, которая принимает на вход состояние среды и возвращает все значения Q(s, a)
  model.compile(loss = loss_function, optimizer = optimizer) #Компилируем модель, используя функцию ошибки, которую объявляем ниже
  return model   #Функция возвращает модель
  
#Инициализируем класс для бинарного дерева
class Priority_Tree:
  data_pointer = 0 #В самом начале инициализируем переменную data_pointer как 0. Позже эта переменная будет показывать нам нужный пример данных по индексу
  
  def __init__(self, memory_capacity):                            #Класс принимает максимальный размер памяти на вход
    self.memory_capacity = memory_capacity                        #Записываем максимальный доступный размер в атрибут класса
    self.tree = np.zeros(2 * memory_capacity - 1)                 #Дерево приоритета инициализируется нулями, но будет заполняться числами. Чем выше число, тем выше приоритет соответствующих данных
    self.data = np.zeros(memory_capacity, dtype = object)         #Этот массив будет содержать данные из игровой среды. У него тип данных object, поскольку он будет содержать разные типы данных в виде кортежей (действия, состояния, и т. д.)
    self.memory_size = 0                                          #Добавим атрибут, который будет показывать нам количество данных в массиве (это не часть алгоритма, используется для удобства)

  #ВАЖНО!!!: размерность tree отличается от размерности data, потому что tree — это целое дерево, и все, кроме последнего уровня этого дерева, существуют лишь для ускоренного поиска 
  #Приоритеты данных хранятся в нижнем слое дерева, то есть с индекса memory_capacity до 2 * memory_capacity - 1 ([memory_capacity : 2*memory_capacity-1])

  #Этот метод будет использоваться при добавлении новых данных в дерево
  #Метод принимает данные, которые нужно добавить, и их приоритет.
  def add(self, data, priority):
    tree_index = self.data_pointer + self.memory_capacity - 1 #Задаем индекс данных в дереве приоритета
    self.data[self.data_pointer] = data                       #Записываем новые данные в индекс, который указывает data_pointer
    self.update(tree_index, priority)                         #Вызываем метод для обновления всех значений по дереву (этот метод объявляется ниже)
    self.data_pointer += 1                                    #Увеличиваем индекс data_pointer на 1
    if self.data_pointer >= self.memory_capacity:             #Если data_pointer больше, чем размер памяти:
      self.data_pointer = 0                                   #Устанавливаем data_pointer как 0 и начинаем его увеличивать с начала (это поможет удалить очень старые данные)
    
    self.memory_size = min(self.memory_size + 1, self.memory_capacity) #Увеличиваем текущий размер памяти на 1, но убеждаемся, что он не больше, чем memory_capacity (это не часть алгоритма, используется для удобства)

  #Объявляем метод для обновления всех приоритетов в дереве. Он вызывается, когда мы добавляем новые данные в дерево
  #Метод принимает на вход текущий индекс дерева и приоритет, по которому нужно обновить
  def update(self, tree_index, priority):
    change = priority - self.tree[tree_index] #Считаем разницу в приоритете (текущий приоритет минус то, что уже есть на этом месте)
    self.tree[tree_index] = priority          #Устанавливаем новый приоритет в лист дерева
    while tree_index != 0:                    #В этом цикле ищем все данные по ветке, пока не достигаем начала дерева
      tree_index = (tree_index - 1)//2        #Берем индекс родительского листа
      self.tree[tree_index] += change         #И увеличиваем его приоритет на параметр change

  #Объявляем метод, который позволит нам извлечь индекс, приоритет и данные, соответствующие одному узлу из дерева
  #Процесс поиска значения в дереве — метод принимает на вход значение приоритета, по которому мы извлекаем пример данных
  def get_leaf(self, value):
    parent_index = 0                              #Индекс родительского узла — инициализируется как 0
    while True:                                   #В цикле ищем нужный нам узел
      left_child_index = 2 * parent_index + 1     #По определению, левый дочерний узел всегда имеет такой индекс по отношению с его родительским узлом
      right_child_index = left_child_index + 1    #По определению, правый дочерний узел всегда имеет индекс равен левому дочернему излу + 1

      if left_child_index >= len(self.tree):      #Если индекс левого дочернего узла, который мы рассматриваем, больше, чем размер дерева, значит мы уже на максимальной глубине дерева
        leaf_index = parent_index                 #Поэтому нужный индекс становится родительским индексом, и мы выходим из цикла
        break
      elif value <= self.tree[left_child_index]:  #Иначе, если значение меньше, чем левый дочерний узел дерева
        parent_index = left_child_index           #Мы записываем левый дочерний индекс как родительский индекс
      else:
        value -= self.tree[left_child_index]      #Иначе, если значение НЕ меньше, чем левый дочерний узел дерева и мы НЕ на максимальной глубине дерева
        parent_index = right_child_index          #родительским индексом становится правый дочерний индекс и значение уменьшается на приоритет левого дочернего узла (по определению, так происходит поиск нужного значения в дереве сумм)

    data_index = leaf_index - self.memory_capacity + 1                #Индекс из массива равен индексу из дерева — размер_памяти + 1
    return leaf_index, self.tree[leaf_index], self.data[data_index]   #Возвращаем индекс узла, приоритет индекса из дерева приоритетов и соответствующие сэмплы данных

  @property                    #Добавляем метод с декоратором свойства
  def total_priority(self):
    return self.tree[0]        #Этот метод возвращает приоритет первого элемента (что по определению — сумма всех остальных приоритетов)


#Дальше создаем класс для воспроизведения приоритетного опыта. В этом классе есть два основных параметра, которые мы называем A и B
class Prioritized_Experience_Replay:
  
  def __init__(self, memory_capacity):                      #Класс принимает на вход максимальный объем памяти для приоритетного дерева
    self.priority_tree = Priority_Tree(memory_capacity)     #Создаем объект приоритетного дерева внутри этого класса
    self.InitialVariables()
  def InitialVariables(self):
    self.PER_A = 0.6      #Задаем значение параметра A
    self.PER_B = 0.4      #Задаем значение параметра B

    self.PER_B_increment_sampling = 1e-6     #Также есть параметр PER_B_increment_sampling, который будет влиять на то, как быстро меняется параметр PER_B во время обучения сети
    self.absolute_error_minimum = 0.01       #Этот параметр отвечает за минимальную ошибку. Если ошибка в предсказании ниже, чем этот параметр, меняем эту ошибку на значения параметра. Это делается для того, чтобы всегда была какая-то вероятность выбора всех данных
    self.absolute_error_maximum = 10.0       #Последний параметр отвечает за максимальную ошибку, которая будет учитываться (т. е. максимально возможный приоритет). Этот параметр нужно подбирать в соответствии величин наград из среды

  #Объявляем метод для сохранения новых данных. Как обычно, данные это — предыдущее состояние, действие, награда, следующее состояние и статус завершения игры
  def store(self, experience):
    maximum_priority = np.max(self.priority_tree.tree[-self.priority_tree.memory_capacity:]) #Смотрим на максимальный приоритет в дереве среди данных в самом низу дерева (у которых нету дочернего узла)

    if maximum_priority == 0:                       #Если максимальный приоритет отсутствует
      maximum_priority = self.absolute_error_maximum  #Мы устанавливаем максимальный приоритет параметром absolute_error_maximum, который задавался выше

    self.priority_tree.add(experience, maximum_priority)  #Затем добавляем данные в дерево поиска вместе с максимальным приоритетом

  #Создаем метод для выбора пакета данных из буфера памяти. Метод принимает на вход размер нужного пакета
  def sample_batch(self, batch_size):
    minibatch = []                       #Выбранные данные будут добавляться в список, который инициализируем под названием minibatch
    batch_index, batch_weights = np.empty((batch_size, ), dtype = np.int32), np.empty((batch_size, 1), dtype=np.float32) #Также создаем пустые NumPy-массивы, где будем хранить индексы данных из пакета и их степень важности
    priority_segment = self.priority_tree.total_priority / batch_size     #У нас будут разные уровни приоритета для данных. Здесь задаем диапазон каждого уровня
    self.PER_B = np.min([1.0, self.PER_B + self.PER_B_increment_sampling])#Увеличиваем параметр PER_B на PER_B_increment_sampling

    p_min = np.min(self.priority_tree.tree[-self.priority_tree.memory_capacity:]) / self.priority_tree.total_priority #Находим минимальный приоритет из датасета и делим его на сумму всех приоритетов
    max_weight = (p_min * batch_size) ** (-self.PER_B)   #Максимальная возможная степень важности — это минимальный приоритет * размер пакета ^ (-1 * PER_B)

    for i in range(batch_size):  #Проходимся циклом, чтобы извлечь все нужные пакеты
      a, b = priority_segment * i, priority_segment * (i + 1)     #Извлекаем верхнюю и нижнюю границы диапазона приоритета
      value = np.random.uniform(a, b)                             #Рандомно генерируем число из диапазона, по которому будем извлекать данные. Это число будет нашим приоритетом
      index, priority, data = self.priority_tree.get_leaf(value)  #Извлекаем индекс, приоритет и данные по ранее заданному числу
      sampling_probabilities = priority / self.priority_tree.total_priority  #Вероятность выбора используется позже при подсчете степени важности данных
      batch_weights[i, 0] = np.power(batch_size * sampling_probabilities, -self.PER_B)/max_weight #Устанавливаем веса так, чтобы данные с более высоким приоритетом имели больше веса (чем больше параметр PER_B, тем больше приоритета будет отдаваться таким данным)
      batch_index[i] = index                        #Извлекаем индекс из буфера памяти
      minibatch.append([data[i] for i in range(5)]) #Добавляем текущее состояние, действие, награду, следующее состояние, состояние завершение игры в список пакета

    return batch_index, minibatch, batch_weights  #Возвращаем индексы пакета, данные пакета и степень важности данных

  #Создаем функцию для обновления всех приоритетов. Этот метод будет вызываться при каждой итерации обучения. Метод принимает на вход индексы (которые возвращаем sample_batch), а также разницы между предсказанными и истинными значениями Q(s, a)
  def batch_update(self, tree_indices, absolute_errors): 
    clipped_errors = np.clip(absolute_errors, self.absolute_error_minimum, self.absolute_error_maximum) #Урезаем все ошибки чтобы они находились в диапазоне между заданным минимумом и максимумом
    normalized_errors = np.power(clipped_errors, self.PER_A) #Возводим ошибки в экспонент PER_A. Чем ниже значение этого параметра, тем равномернее будет приоритет
    
    for i, j in zip(tree_indices, normalized_errors): #Проходимся по индексам и нормированным ошибкам
      self.priority_tree.update(i, j)                 #Обновляем дерево приоритетов в соответствии с данными

  @property                #Для удобства добавляем метод с декоратором свойства, который будет возвращать размер заполненной памяти
  def buffer_size(self):
    return self.priority_tree.memory_size
class ModelTrainer():
    def __init__(self,TrainSettings,Models,Env,load_pretrained=True):                      #Класс принимает на вход максимальный объем памяти для приоритетного дерева
        self.gamma,self.initial_epsilon,self.final_epsilon,self.epsilon_decay_factor,self.observation_steps,self.target_model_update_rate,self.learning_rate,self.batch_size,self.memory_capacity,self.TrainUpdateRate=TrainSettings

        self.memory_buffer = Prioritized_Experience_Replay(self.memory_capacity)
        self.record_rewards = []
        self.episode_number=1
        self.timestep = 0
        self.main_model=Models[0]
        self.target_model=Models[1]
        self.Env=Env
        self.action_size=len(self.Env.ActionSpace)
        self.Statistic=[]
        self.memory_buffer.InitialVariables()
        self.record_rewards=[]
        self.episode_number=0
        self.timestep=0
        self.epsilon=1
        if(load_pretrained and os.path.isfile('dueling_qn_main2.h5')):
            LoadState()
    def LoadState(self):
        with open('dueling_DQN_stats2.txt', 'rb') as f:  #Записываем статистику в файл через библиотеку pickle
                self.record_rewards, self.episode_number,self.Statistic, self.timestep, self.epsilon, self.memory_buffer.PER_B=pickle.load(f) 


    def SetTrainState(self,record_rewards, episode_number, timestep, epsilon, PER_B):
        self.memory_buffer.InitialVariables()
        self.record_rewards=record_rewards
        self.episode_number=episode_number
        self.timestep=timestep
        self.epsilon=epsilon
        self.memory_buffer.PER_B=PER_B

    def get_action(self,state, epsilon, action_size):

      if random.random() <= self.epsilon:  #Генерируем рандомное значение, если оно меньше или равно эпсилону, берем рандомное действие
        action_index = np.random.randint(0, action_size) #Иными словами, мы берем рандомное действие с вероятностью эпсилон
      else: #Иначе (наше рандомное число больше, чем эпсилон)
        #print("get_action",state[0].shape,state[1].shape)
        Q_values = self.main_model.predict(state, verbose = 0) #models — название переменной, которая будет содержать целевую и основную модели (объект класса Models_Class, который создается выше)
        action_index = np.argmax(Q_values) #Извлекаем индекс действия, который приводит к максимальному значению Q(s, a)

      if self.memory_buffer.buffer_size >= self.observation_steps:  #Снижаем значение эпсилон, если буфер памяти достаточно большой (идет обучение) и epsilon больше, чем final_epsilon, снижаем значение epsilon на epsilon_decay_factor
        epsilon = max(epsilon * self.epsilon_decay_factor, self.final_epsilon) #Снижаем значение эпсилон умножением (это приведет к экспоненциальному спаду). Убеждаемся, что значение эпсилон не ниже, чем final_epsilon

      return action_index, epsilon #Возвращаем выбранное действие и новое значение epsilon
    
    def train_network(self):
      tree_index, replay_samples, sample_weights = self.memory_buffer.sample_batch(self.batch_size) #В начале мы собираем данные из приоритизированного буфера памяти
      #(previous_state, action_index, reward, current_state, episode_done)
      ShapeState1=replay_samples[0][0][0].shape
      #ShapeState2=replay_samples[0][0][1].shape
      #print("TrainStart")
      previous_states1 = np.zeros(((self.batch_size,) + ShapeState1)) #Создаем массив из нулей, где будем хранить предыдущие состояния
      current_states1 = np.zeros(((self.batch_size,) + ShapeState1))  #Создаем массив из нулей, где будем хранить следующие состояния
      previous_states2 = np.zeros(((self.batch_size,) + (1,))) #Создаем массив из нулей, где будем хранить предыдущие состояния
      current_states2 = np.zeros(((self.batch_size,) + (1,)))  #Создаем массив из нулей, где будем хранить следующие состояния
      #print(previous_states1.shape,previous_states2.shape)
      actions, rewards, done = [], [], []      #Инициализируем действия, награды и состояния завершения игры пустыми списками
      for i in range(self.batch_size):              #Проходимся по собранному пакету данных
        previous_states1[i,:,:] = replay_samples[i][0][0] #Собираем все предыдущие состояния в массив
        previous_states2[i,:] = replay_samples[i][0][1] #Собираем все предыдущие состояния в массив
        actions.append(replay_samples[i][1])            #Собираем все действия
        rewards.append(replay_samples[i][2])            #Собираем все награды
        current_states1[i,:,:] = replay_samples[i][3][0]  #Собираем все следующие состояния в массив
        current_states2[i,:] = replay_samples[i][3][1]  #Собираем все следующие состояния в массив
        done.append(replay_samples[i][4])               #Собираем все статусы завершения игры 

      #print("DS complite")
      Q_values = self.main_model.predict([previous_states1,previous_states2],verbose=0)        #С начала предсказываем Q(s, a) из основной модели
      target_Q_values = self.target_model.predict([current_states1,current_states2],verbose=0)  #В конце предсказываем Q(s', a') из целевой модели

      old_Q_values = Q_values.copy()     #Чтобы использовать приоритизированный буфер памяти, нам нужно будет посчитать разницу между истинными и предсказанными значениями Q(s, a), поэтому здесь запоминаем предсказанные значения

      for i in range(self.batch_size):        #Итерируем через пакет данных, как и в предыдущем занятии по Q-learning
        if done[i]:                             
          Q_values[i,actions[i]] = rewards[i]  #Если флажок done равен True, значит это последнее состояние в эпизоде, и его награда равна награде, которую выдала игровая среда
        else:
          Q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(target_Q_values[i, :]) #Новые Q-значения устанавливаются в соответствии с уравнением Беллмана

      indices = np.arange(self.batch_size, dtype=np.int32) #Задаем индексы всех собранных данных (используется в следующей строке кода)
      absolute_errors = np.abs(old_Q_values[indices, np.array(actions)] - Q_values[indices, np.array(actions)]) #Считаем абсолютную ошибку между предсказанными и истинными Q-значениями, что будет использоваться в подсчете приоритета
      loss = self.main_model.fit([previous_states1,previous_states2], Q_values, batch_size = self.batch_size, sample_weight = sample_weights, verbose=0) #Обучаем модель на состояниях и Q-значениях с учетом sample_weights (веса важности разных данных) в итоговой ошибке
      #print("TrainEnd")
      self.memory_buffer.batch_update(tree_index, absolute_errors) #Обновляем дерево приоритетов для всего пакета данных

      if self.timestep % self.target_model_update_rate == 0: #Поскольку у нас алгоритм Double Dueling Deep-Q Network, мы будем обновлять веса целевой модели из основной модели раз в заданное количество шагов
        self.target_model.set_weights(self.main_model.get_weights())
    def train_cycle(self):
        self.Statistic=[]
        ProfitCum,PFCum=0,0
        self.Env.reset()                       #Начинаем новый эпизод игры
        current_state = self.Env.get_state()             #Извлекаем информацию о текущем состоянии игры
        current_info = self.Env.get_variables()  #current_info — массив, который содержит количество убитых врагов, наличие патронов и здоровье на текущий момент
        previous_info = current_info.copy()             #previous_info — массив, который содержит количество убитых врагов, наличие патронов и здоровье на предыдущий момент
        previous_state=current_state
        #current_state1 = current_state[0]    #В качестве первого состояния просто дублируем кадр 8 раз   
        #current_state2 = current_state[1]    #В качестве первого состояния просто дублируем кадр 8 раз   
        #previous_state1 = current_state1  #Инициализируем предыдущий шаг как текущий шаг
        #previous_state2 = current_state2  #Инициализируем предыдущий шаг как текущий шаг
        interval_reward=0
        TrainCycleCnt=0

        
        while self.episode_number < 15001:
          self.timestep += 1
          #print(previous_state[0].shape,previous_state[1])
          previous_state1_1=np.expand_dims(previous_state[0], axis = 0)
          previous_state2_1=np.expand_dims(previous_state[1], axis = 0)
          if(len(previous_state2_1.shape)<2):
              previous_state2_1=np.expand_dims(previous_state2_1, axis = 0)
          #print("PS",previous_state1_1.shape,previous_state2_1.shape)
          #action_index, self.epsilon = self.get_action([previous_state1_1,previous_state2_1], self.epsilon, self.action_size)  #Извлекаем индекс награды и новое значение эпсилон
          action_index, self.epsilon = self.get_action([previous_state1_1,previous_state2_1], self.epsilon, self.action_size)  #Извлекаем индекс награды и новое значение эпсилон
          #action_onehot = keras.utils.to_categorical(action_index)                              #Приводим награду в onehot-массив
          #print(action_index, self.epsilon,action_onehot)
          current_state,reward,episode_done=self.Env.Step(self.Env.ActionSpace[action_index])                                   #Подаем действие в игровую среду в качестве списка 
          Profit,PF=self.Env.get_variables()
          
          if episode_done: #Нам необходимо возобновить среду и записать нужные статистики, когда заканчивается эпизод
            self.episode_number += 1   #Увеличиваем номер эпизода на 1
            
            #Чтобы не собирать слишком много данных и чтобы данные было удобно отображать на графике, мы записываем результаты лишь раз в 10 эпизодов
            if self.episode_number % 10 == 0 and self.episode_number > 2: #Записываем результат раз в 10 эпизодов
              self.record_rewards.append(interval_reward)            #Добавляем награду в список всех наград
              self.Statistic.append([ProfitCum/10,PFCum/10])
              interval_reward=0
              ProfitCum=0
              PFCum=0
              #show_scores(record_rewards, record_kills)  #Записываем результаты в графики
              self.main_model.save_weights('dueling_qn_main2.h5') #Сохраняем веса основной модели
              self.target_model.save_weights('dueling_qn_target2.h5') #Сохраняем веса целевой модели
              with open('dueling_DQN_stats2.txt', 'wb') as f:  #Записываем статистику в файл через библиотеку pickle
                pickle.dump([self.record_rewards, self.episode_number,self.Statistic, self.timestep, self.epsilon, self.memory_buffer.PER_B], f) 
              print("Статистика успешно сохранена.")
              #print(environment_reward,current_info)
          
            print(f"Закончился {self.episode_number}-й эпизод. Значение эпсилон: {round(self.epsilon, 2)}, SummReward: {Profit}, PF: {PF}")


            self.Env.reset()                       #Затем необходимо начать новый эпизод игры
            current_state = self.Env.get_state()            #Извлекаем новое состояние игры

          current_info = self.Env.get_variables()  #Извлекаем информацию об игровой среде (количество убитых врагов, неиспользованных патронов, текущее здоровье)
          environment_reward = self.Env.get_last_reward()                            #Извлекаем награду за шаг из игровой среды
          custom_reward = 0#get_reward(previous_info, current_info, episode_done)  #Извлекаем нагаду за шаг, используя самописную логику
          reward = environment_reward + custom_reward  #Награда за действие будет суммой награды из игровой среды и самописной награды

          interval_reward += reward #Добавляем награду в переменную для статистики
          ProfitCum+=Profit
          PFCum+=PF
          #reward = reward/50        #Нормируем награду делением на 50
          self.memory_buffer.store((previous_state, action_index, reward, current_state, episode_done)) #Добавляем предыдущее состояние, действие, награду и текущее состояние в память
          #print("train",self.memory_buffer.buffer_size)
          if self.memory_buffer.buffer_size >= self.observation_steps and TrainCycleCnt>self.TrainUpdateRate: #Если у нас достаточно данных в буфере памяти для обучения алгоритма:
            self.train_network()   #Обучаем модели по очереди, используя самописный метод
            TrainCycleCnt=0
            #print("train",self.memory_buffer.buffer_size)

          TrainCycleCnt+=1
          previous_info = current_info    #Запоминаем предыдущую информацию
          previous_state = current_state  #Запоминаем предыдущее состояние
    
def LoadFromMultishareToDF(FileName:str="",StartYear:int=0,ShareNumber:int=0,TrainPart:float=0.9):
    x_data_1 = np.loadtxt(FileName)
    #print(x_data_1[:3,0],StartYear,x_data_1.shape,type(StartYear))
    #print(len(x_data_1[x_data_1[:,0]>=np.int64(StartYear)]))
    x_data_1=x_data_1[np.where(x_data_1[:,0]>=StartYear)[0][0]:]
    
    x_data_tr=x_data_1[:int(len(x_data_1)*TrainPart)]
    x_data_test=x_data_1[int(len(x_data_1)*TrainPart):]
    x_time_tr=np.zeros(len(x_data_tr),dtype=datetime)
    x_time_test=np.zeros(len(x_data_test),dtype=datetime)
    for i in range(len(x_time_tr)):
        x_time_tr[i]=datetime(int(x_data_tr[i,0]),int(x_data_tr[i,1]),int(x_data_tr[i,2]),int(x_data_tr[i,5]),int(x_data_tr[i,6]))
    for i in range(len(x_time_test)):
        x_time_test[i]=datetime(int(x_data_test[i,0]),int(x_data_test[i,1]),int(x_data_test[i,2]),int(x_data_test[i,5]),int(x_data_test[i,6]))
    x_time_tr=np.array(x_time_tr)
    x_time_test=np.array(x_time_test)
    DFDictTr={"Date":x_time_tr,"Open":x_data_tr[:,7+ShareNumber*5+0],"High":x_data_tr[:,7+ShareNumber*5+1],
              "Low":x_data_tr[:,7+ShareNumber*5+2],"Close":x_data_tr[:,7+ShareNumber*5+3],
              "Adj Close":x_data_tr[:,7+ShareNumber*5+3],"Volume":x_data_tr[:,7+ShareNumber*5+4]}
    DFTrain=pd.DataFrame(DFDictTr).set_index("Date")
    del DFDictTr
    DFDictTest={"Date":x_time_test,"Open":x_data_test[:,7+ShareNumber*5+0],"High":x_data_test[:,7+ShareNumber*5+1],
              "Low":x_data_test[:,7+ShareNumber*5+2],"Close":x_data_test[:,7+ShareNumber*5+3],
              "Adj Close":x_data_test[:,7+ShareNumber*5+3],"Volume":x_data_test[:,7+ShareNumber*5+4]}
    DFTest=pd.DataFrame(DFDictTest).set_index("Date")
    del DFDictTest
    return DFTrain,DFTest
