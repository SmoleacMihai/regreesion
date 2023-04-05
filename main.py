import pandas as pd
import numpy as np  # not used in this task
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Lasso


if __name__ == '__main__':
    # Загрузка данных
    train_df = pd.read_excel('price.xlsx')
    test_df = pd.read_excel('predict.xlsx')

    # Обучение моделей
    linear_reg = LinearRegression()
    lasso_reg = Lasso(alpha=0.1)

    linear_reg.fit(train_df[['area']], train_df.price)
    lasso_reg.fit(train_df[['area']], train_df.price)

    # Предсказание цен на квартиры из файла predict.xlsx
    test_df['linear_reg_price'] = linear_reg.predict(test_df[['area']])
    test_df['lasso_reg_price'] = lasso_reg.predict(test_df[['area']])

    # Сохранение результатов в новые файлы
    test_df[['area', 'linear_reg_price']].to_excel('predicted_values_ap1.xlsx', index=False)
    test_df[['area', 'lasso_reg_price']].to_excel('predicted_values_ap2.xlsx', index=False)