import numpy as np


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Невозможно умножить матрицы: число столбцов первой не равно числу строк второй")

    rows_a = len(matrix_a)
    cols_b = len(matrix_b[0])
    cols_a = len(matrix_a[0])

    result = [[0] * cols_b for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    """
    coeffs1 = list(map(float, a_1.split()))
    coeffs2 = list(map(float, a_2.split()))

    # Находим точки экстремума
    x_ext1 = -coeffs1[1] / (2 * coeffs1[0]) if coeffs1[0] != 0 else None
    x_ext2 = -coeffs2[1] / (2 * coeffs2[0]) if coeffs2[0] != 0 else None

    # Если функции совпадают - решений бесконечно много
    if coeffs1 == coeffs2:
        return None

    # Решаем уравнение: f(x) = p(x) -> a1*x^2 + b1*x + c1 = a2*x^2 + b2*x + c2
    a = coeffs1[0] - coeffs2[0]
    b = coeffs1[1] - coeffs2[1]
    c = coeffs1[2] - coeffs2[2]

    # a*x^2 + b*x + c = 0
    if a == 0:
        if b == 0:
            return None if c == 0 else []  # Бесконечно много решений или их нет
        return [(-c / b, np.polyval(coeffs1, -c / b))]  # Линейное уравнение

    # Квадратное уравнение
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return []  # Нет решений
    elif discriminant == 0:
        x = -b / (2 * a)
        return [(x, np.polyval(coeffs1, x))]  # Одно решение
    else:
        x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        x2 = (-b - np.sqrt(discriminant)) / (2 * a)
        return [(x1, np.polyval(coeffs1, x1)), (x2, np.polyval(coeffs1, x2))]


def skew(x):
    """
    Функция для расчета коэффициента асимметрии.
    Возвращает значение, округленное до 2 знаков после запятой.
    """
    x_mean = np.mean(x)
    x_std = np.std(x, ddof=1)
    m3 = np.mean((x - x_mean) ** 3)
    skewness = m3 / (x_std ** 3)
    return round(skewness, 2)


def kurtosis(x):
    """
    Функция для расчета коэффициента эксцесса.
    Возвращает значение, округленное до 2 знаков после запятой.
    """
    x_mean = np.mean(x)
    x_std = np.std(x, ddof=0)
    m4 = np.mean((x - x_mean) ** 4)
    excess_kurtosis = (m4 / (x_std ** 4)) - 3
    return round(excess_kurtosis, 2)
