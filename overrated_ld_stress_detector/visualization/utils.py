import plotly.express as px
import numpy as np


def visualize_data(df,
                   user_id,
                   test_id,
                   presentation_id,
                   result=None,
                   frequency=20,
                   len_seconds=12):
    """
    Visualize data.
    df - pandas dataframe
    user_id - user id
    test_id - test id
    presentation_id - presentation id
    """

    vals = df[df.Filename == f'{user_id}'].query(f'Test_index=={test_id}').query(
        f'Presentation=={presentation_id}').Data.apply(eval)
    res = [val for val in vals if len(val) == frequency * len_seconds]
    data = np.array(res)

    vals2 = df[df.Filename == f'{user_id}'].query(f'Test_index=={test_id}').query(
        f'Presentation=={presentation_id}').Data_2.apply(eval)
    res2 = [val for val in vals2 if len(val) == frequency * len_seconds]
    data2 = np.array(res2)

    if result is None:
        target = df[df.Filename == f'{user_id}'].query(f'Test_index=={test_id}').query(
        f'Presentation=={presentation_id}').Class_label.values
    else:
        target = result

    col_dict = {0: 'green', 1: 'yellow', 2: 'red'}

    fig = px.line(y=data.flatten())
    fig.update_layout(title=f'Фотоплетизмограмма для всех вопросов пользователю {user_id} '
                            f'для группы вопросов {test_id} с {presentation_id} номером повторения.\n',
                      legend_title='Классы вопросов',
                      xaxis_title='Время',
                      yaxis_title='Амплитуда сигнала',
                      )
    fig.update_layout()

    for i in range((frequency * len_seconds), len(data.flatten()) + 1, frequency * len_seconds):
        fig.add_vrect(x0=i - frequency * len_seconds,
                      x1=i,
                      fillcolor=col_dict[target[i // (frequency * len_seconds) - 1]],
                      opacity=0.2,
                      annotation={"text": str(int(target[i // (frequency * len_seconds) - 1])),
                                  "showarrow": False},
                      )


    fig.show()

    fig = px.line(y=data2.flatten())
    fig.update_layout(title=f'Пьезоплетизмограмма для всех вопросов пользователю {user_id}'
                            f'для группы вопросов {test_id} с {presentation_id} номером повторения',
                      xaxis_title='Время',
                      yaxis_title='Амплитуда сигнала')
    for i in range((frequency * len_seconds), len(data2.flatten()) + 1, frequency * len_seconds):
        fig.add_vrect(x0=i - frequency * len_seconds,
                      x1=i,
                      fillcolor=col_dict[target[i // (frequency * len_seconds) - 1]],
                      opacity=0.2,
                      annotation={"text": str(int(target[i // (frequency * len_seconds) - 1])),
                                  "showarrow": False})
    fig.show()
