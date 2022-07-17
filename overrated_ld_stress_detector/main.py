import pandas as pd

from ml import pretrained_model
from visualization import utils

df = pd.read_excel(r"models/dataset_train.xlsx", engine='openpyxl')

utils.visualize_data(df,
                     user_id='8fc79c7f-bbdb-4512-b460-c75aacd1a3c7',
                     test_id=4,
                     presentation_id=1)

df = df.drop("Class_label",
             axis=1)

model = pretrained_model.get_model(model_type='cnn')

result = model.predict(df.iloc[:200])
print(result)
utils.visualize_data(df,
                     user_id='8fc79c7f-bbdb-4512-b460-c75aacd1a3c7',
                     test_id=3,
                     presentation_id=1,
                     result=result)
