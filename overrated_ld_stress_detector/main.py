import pandas as pd

from ml import pretrained_model
from visualization import utils

df = pd.read_excel(r"C:\Users\Sergey\Downloads\Telegram Desktop\dataset_train.xlsx", engine='openpyxl')

utils.visualize_data(df,
                     user_id='8fc79c7f-bbdb-4512-b460-c75aacd1a3c7',
                     test_id=4,
                     presentation_id=1)
df = df.drop("Class_label",
             axis=1)

model = pretrained_model.get_model('models/')
result = model.predict(df)

utils.visualize_data(df,
                     user_id='8fc79c7f-bbdb-4512-b460-c75aacd1a3c7',
                     test_id=4,
                     presentation_id=1,
                     result=result)
