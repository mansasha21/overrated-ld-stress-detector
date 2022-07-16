import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def process_data(df):
    func = lambda x: 1 if len(x) > 0 else np.NAN
    try_df = df.copy()
    try_df['Data'] = try_df['Data'].apply(eval)
    try_df['Data_2'] = try_df['Data_2'].apply(eval)

    try_df['check'] = try_df['Data'].apply(func)
    try_df = try_df.dropna()
    try_df['check'] = try_df['Data_2'].apply(func)
    try_df = try_df.dropna().drop(['check'], axis=1).reset_index(drop=True)

    photo = try_df.drop(['Data_2'], axis=1)
    piezo = try_df.drop(['Data'], axis=1)

    scaler_photo = MinMaxScaler()
    flag_photo = True
    for i, row in photo.iterrows():
        try:
            if flag_photo:
                photo.loc[i, 'Data'] = scaler_photo.fit_transform(np.array(row['Data']).reshape(-1, 1))
                flag_photo = False
            else:
                photo.loc[i, 'Data'] = scaler_photo.transform(np.array(row['Data']).reshape(-1, 1))
        except:
            continue

    scaler_piezo = MinMaxScaler()
    flag_piezo = True
    for i, row in piezo.iterrows():
        try:
            if flag_piezo:
                piezo.loc[i, 'Data_2'] = scaler_piezo.fit_transform(np.array(row['Data_2']).reshape(-1, 1))
                flag_piezo = False
            else:
                piezo.loc[i, 'Data_2'] = scaler_piezo.transform(np.array(row['Data_2']).reshape(-1, 1))
        except:
            continue

    all_df = photo
    all_df['Data_2'] = piezo['Data_2']

    percent_df = all_df.copy()
    quant_list = [0.01, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.99]

    def _quant(row: pd.Series, quant=None):
        quant = np.quantile(row, quant)
        return quant

    for q in quant_list:
        try:
            percent_df[f'Data_q_{q}'] = percent_df['Data'].apply(_quant, quant=q)
            percent_df[f'Data2_q_{q}'] = percent_df['Data_2'].apply(_quant, quant=q)
        except:
            continue

    stats_df = percent_df.copy()
    for f,v in zip([np.mean, np.max, np.min, np.median], ["mean", "max", "min", "median"]):
        try:
            stats_df[f'Data_{v}'] = stats_df['Data'].apply(f)
            stats_df[f'Data2_{v}'] = stats_df['Data_2'].apply(f)
        except:
            continue

    prep_df = stats_df.astype({
        'Test_index': 'int',
        'Presentation': 'int',
        'Question': 'int'
    })

    N = 10

    def abs_amplitude(row: pd.Series):
        """
        Модуль амплитуды
        :param row:
        :return:
        """
        F = np.fft.rfft(row, n=None, axis=-1)
        A = [((F[i].real) ** 2 + (F[i].imag) ** 2) ** 0.5 for i in np.arange(0, N, 1)]
        return A

    def phase(row: pd.Series):
        """
        Фаза
        :param row:
        :param F: прямое преобразование Фурье в частотную область
        :return:
        """
        F = np.fft.rfft(row, n=None, axis=-1)
        arg = []
        for i in np.arange(0, N, 1):
            if F[i].imag != 0:
                t = (-np.tanh((F[i].real) / (F[i].imag)))
                arg.append(t)
            else:
                arg.append(np.pi / 2)
        return arg

    prep_df['data_apm'] = prep_df['Data'].apply(abs_amplitude)
    prep_df['data_2_apm'] = prep_df['Data_2'].apply(abs_amplitude)
    prep_df['data_phase'] = prep_df['Data'].apply(phase)
    prep_df['data_2_phase'] = prep_df['Data_2'].apply(phase)
    sensor_1 = prep_df['data_2_apm'].apply(pd.Series)
    sensor_1.columns = [f'data_2_amp_{i}' for i in range(sensor_1.shape[1])]
    sensor_2 = prep_df['data_2_phase'].apply(pd.Series)
    sensor_2.columns = [f'data_2_phase_{i}' for i in range(sensor_2.shape[1])]

    sensor_3 = prep_df['data_apm'].apply(pd.Series)
    sensor_3.columns = [f'data_amp_{i}' for i in range(sensor_3.shape[1])]
    sensor_4 = prep_df['data_phase'].apply(pd.Series)
    sensor_4.columns = [f'data_phase_{i}' for i in range(sensor_4.shape[1])]

    sensor_df = pd.concat([prep_df, sensor_1, sensor_2, sensor_3, sensor_4], axis=1)
    try_set = sensor_df.drop(['Data_2', 'Data', 'Filename',
                              'data_2_apm', 'data_2_phase', 'data_apm', 'data_phase'], axis=1)
    return try_set
