import pandas as pd

from ml import pretrained_model
from visualization import utils

is_test = True
is_visualize = True
if is_test:
    df = pd.read_csv('C:/Data/dataset_test.csv',
                     index_col=0,
                     sep=';')
    if is_visualize:
        utils.visualize_data(df,
                             user_id='8fc79c7f-bbdb-4512-b460-c75aacd1a3c7',
                             test_id=4,
                             presentation_id=1)
else:
    df = pd.read_excel("C:/Data/dataset_train.xlsx",  # путь до данных, на которых необходимо обучиться
                       engine='openpyxl')  # Загрузка данных из Excel

df = df.drop("Class_label",
             axis=1)

model = pretrained_model.get_model(model_type='catboost') # 2 type of models: 'catboost' and 'cnn'

if is_test:
    result = model.predict(df)

    df['Class_label'] = result
    df.to_csv('C:/Data/Overrated.csv',
              sep=';')
    if is_visualize:
        utils.visualize_data(df,
                             user_id='dff4a3a1-3d58-4072-a903-e5b38e2c541f',
                             test_id=3,
                             presentation_id=1,
                             result=result)
else:
    model.train()
