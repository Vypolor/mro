import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import show
from numpy.linalg import det, inv
from scipy.special import erfinv, erf


def generate_standard_normal(n, N, step=50):
    vector = np.zeros([n,N])
    for i in range(0, step):
        vector = vector + [np.random.uniform(-0.5, 0.5, N), np.random.uniform(-0.5, 0.5, N)]
    return vector / (np.sqrt(step) * np.sqrt(1 / 12))


def generate_vector(M, B, ksi): # пункт 1
    a = np.zeros((2, 2))
    x = np.zeros((n, N))

    a[0, 0] = np.sqrt(B[0, 0])
    a[0, 1] = 0
    a[1, 0] = B[0, 1] / np.sqrt(B[0, 0])
    a[1, 1] = np.sqrt(B[1, 1] - B[0, 1] ** 2 / B[0, 0])

    #print (a)

    a = np.matmul(a, ksi)
    x = [a[0] + M[0],
         a[1] + M[1]]
    return np.reshape(x, (2,200))


def estimate_math_expectation(x, N, n): #оценка мат ожидания
    estimate = np.zeros((2, 1))

    for i in range(0, n):
        for j in range(0, N):
            estimate[i, :] += x[i, j] #стр 5 1 формула

    return estimate / N

def correlation_matrix_estimate(x, M, N, n): # оценка корреляционной матрицы
    estimate = np.dot(x, x.transpose())
    return estimate / N - np.dot(M, M.transpose()) # стр 5 2 формула

def Bhatacharya(M1, M2, B1, B2): #cтр 6 (1)
    return (0.25 * np.dot(np.dot((M1 - M2).transpose(), np.linalg.inv((B1 + B2) / 2)), (M1 - M2))
            + 0.5 * math.log(np.linalg.det((B1 + B2) / 2) / np.sqrt(np.linalg.det(B1) * np.linalg.det(B2))))

def Mahalanobis(M1, M2, B): #cтр 6 (2)
    return np.dot(np.dot((M1 - M2).transpose(), np.linalg.inv(B)), (M1 - M2))


if __name__ == '__main__':
    n = 2
    N = 200

    M1 = np.array([[-1], [1]])
    M2 = np.array([[0], [-1]])
    M3 = np.array([[1], [0]])

#кор матрица
    B1 = np.matrix(([0.1, 0],
                    [0, 0.1]))
    B2 = np.matrix(([0.25, 0.03]
                  , [0.03, 0.25]))
    B3 = np.matrix(([0.2, -0.2],
                    [-0.2, 0.25]))

    ksi1 = generate_standard_normal(n, N)
    ksi2 = generate_standard_normal(n, N)
    ksi3 = generate_standard_normal(n, N)

    Y1 = generate_vector(M1, B2, ksi1)
    Y2 = generate_vector(M2, B2, ksi2)

    X1 = generate_vector(M1, B1, ksi1)
    X2 = generate_vector(M2, B2, ksi2)
    X3 = generate_vector(M3, B3, ksi3)

    plt.scatter(np.array(Y1[0, :]), np.array(Y1[1, :]), color='red')
    plt.scatter(np.array(Y2[0, :]), np.array(Y2[1, :]), color='blue')
    plt.show()

    plt.scatter(np.array(X1[0, :]), np.array(X1[1, :]), color='red')
    plt.scatter(np.array(X2[0, :]), np.array(X2[1, :]), color='green')
    plt.scatter(np.array(X3[0, :]), np.array(X3[1, :]), color='blue')
    plt.show()


    m1 = estimate_math_expectation(Y1, N, n)
    m2 = estimate_math_expectation(Y2, N, n)
    m3 = estimate_math_expectation(X1, N, n)
    m4 = estimate_math_expectation(X2, N, n)
    m5 = estimate_math_expectation(X3, N, n)


    print('\nОценка математического ожидания:\nМ1 по Y1:', m1, 'М2 по Y2:', m2, 'М1 по X1:', m3, 'М2 по X2:', m4, 'М3 по X3:', m5, sep='\n')

    print('\nОценка корреляционной матрицы:\nB2 по Y1:', correlation_matrix_estimate(Y1, m1, N, n), 'B2 по Y2:', correlation_matrix_estimate(Y2, m2, N, n),
          'B1 по X1:', correlation_matrix_estimate(X1, m3, N, n), 'В2 по X2:', correlation_matrix_estimate(X2, m4, N, n), 'В3 по X3:', correlation_matrix_estimate(X3, m5, N, n),
          sep='\n')

    print('\nРасстояние Бхатачария 1-2: ', Bhatacharya(m1, m2, B1, B2), '\nРасстояние Бхатачария 2-3: ',
          Bhatacharya(M2, M3, B2, B3),
          '\nРасстояние Бхатачария 1-3: ', Bhatacharya(M1, M3, B1, B3))

    print('\nРасстояние Махаланобиса: ', Mahalanobis(M1, M2, B1))

    np.savetxt("Y1.csv", Y1, delimiter='|')
    np.savetxt("Y2.csv", Y2, delimiter='|')
    np.savetxt("X1.csv", X1, delimiter='|')
    np.savetxt("X2.csv", X2, delimiter='|')
    np.savetxt("X3.csv", X3, delimiter='|')


    #============================================================================================================
    #============================================================================================================
    #================================================ 2 LABA ====================================================
    #============================================================================================================
    #============================================================================================================


    def estimate_math_expectation(x, N, n):  # Оценка мат ожидания
        estimate = np.zeros((2, 1))

        for i in range(0, n):
            for j in range(0, N):
                estimate[i, :] += x[i, j]

        return estimate / N


    def correlation_matrix_estimate(x, M, N, n):  # Оценка корреляционной матрицы
        estimate = np.dot(x, x.transpose())
        return estimate / N - np.dot(M, M.transpose())


    def byesian_boundary(M1, M2, X, B, P1, P2):  # 13 страница случай 2 предпоследняя формула похожа.
        Y = np.array([0])

        r = 1 / 2 * np.transpose(M1 + M2) * inv(np.matrix(B)) * (M1 - M2) + np.log(P1 / P2)
        l1 = (np.transpose(M1 - M2) * inv(np.matrix(B)))[0, 0]
        l2 = (np.transpose(M1 - M2) * inv(np.matrix(B)))[0, 1]

        for x in X:
            # todo:UNCLEAR
            Y = np.append(Y, float((r - l1 * x) / l2))

        return Y[1:]


    def Mahalanobis(M1, M2, B):
        return np.dot(np.dot((M1 - M2).transpose(), np.linalg.inv(B)), (M1 - M2))


    def probability_erroneous_classification(M1, M2, B):  # вероятность ошибочной классификации 19 страница
        return (0.5 - 0.5 * erf(1 / 2 * np.sqrt(Mahalanobis(M1, M2, B))),
                0.5 + 0.5 * erf(-1 / 2 * np.sqrt(Mahalanobis(M1, M2, B))))


    def task_one(Y1, Y2):
        M1 = estimate_math_expectation(Y1, 200, 2)
        M2 = estimate_math_expectation(Y2, 200, 2)
        B = correlation_matrix_estimate(Y1, M1, 200, 2)
        X = np.linspace(-1, 2, 100)

        Y = byesian_boundary(M1, M2, X, B, 0.5, 0.5)

        fig = plt.figure(figsize=(5, 3))
        fig.add_subplot(1, 1, 1)
        plt.plot(X, Y, color='yellow', linewidth=3)
        plt.title('Байесовская граница')
        plt.scatter(X1[0][:], X1[1][:], c='blue')
        plt.scatter(X2[0][:], X2[1][:], c='red')
        show()

        p1, p2 = probability_erroneous_classification(M1, M2, B)
        p = p1 + p2
        print("1 вероятность ошибочной классификации: ", float(p1))
        print("2 вероятность ошибочной классификации: ", float(p2))
        print("Суммарная вероятность ошибочной классификации: ", float(p))


    def Neiman_Pearson(X, M1, M2, B, p0):  # страница 19 в самом конце формула.
        Y = np.array([0])
        L = np.exp(-.5 * Mahalanobis(M1, M2, B) + np.sqrt(Mahalanobis(M1, M2, B)) * erfinv(1 - p0))

        r = 1 / 2 * np.transpose(M1 + M2) * inv(np.matrix(B)) * (M1 - M2) + np.log(L)
        l1 = (np.transpose(M1 - M2) * inv(np.matrix(B)))[0, 0]
        l2 = (np.transpose(M1 - M2) * inv(np.matrix(B)))[0, 1]

        for x in X:
            Y = np.append(Y, float((r - l1 * x) / l2))

        return Y[1:]


    def min_max_boundary(X, M1, M2, B):  # как байесовский, только добавляем минимальную ошибку
        Y = np.array([0])
        l_min_max = 0

        r = 1 / 2 * np.transpose(M1 + M2) * inv(np.matrix(B)) * (M1 - M2) + l_min_max
        l1 = (np.transpose(M1 - M2) * inv(np.matrix(B)))[0, 0]
        l2 = (np.transpose(M1 - M2) * inv(np.matrix(B)))[0, 1]

        for x in X:
            Y = np.append(Y, float((r - l1 * x) / l2))

        return Y[1:]


    def task_two(Y1, Y2):

        M1 = estimate_math_expectation(Y1, 200, 2)
        M2 = estimate_math_expectation(Y2, 200, 2)
        B = correlation_matrix_estimate(Y1, M1, 200, 2)
        X = np.linspace(-1, 2, 100)

        X1 = Neiman_Pearson(X, M1, M2, B, 0.05)
        X2 = min_max_boundary(X, M1, M2, B)

        fig = plt.figure(figsize=(8, 3))
        fig.add_subplot(1, 2, 1)
        plt.plot(X, X1, color='yellow', linewidth=5)
        plt.title('Неймана Пирсона')
        plt.scatter(Y1[0][:], Y1[1][:], c='blue')
        plt.scatter(Y2[0][:], Y2[1][:], c='red')
        fig.add_subplot(1, 2, 2)
        plt.plot(X, X2, color='yellow', linewidth=5)
        plt.title('Минимаксный')
        plt.scatter(Y1[0][:], Y1[1][:], c='blue')
        plt.scatter(Y2[0][:], Y2[1][:], c='red')
        show()


    def byesian_boundary_diff(X, M1, M2, B1, B2, P1, P2):  # страница 13 в конце
        X1 = []
        for y in X:
            c_x = (2 * (np.transpose(M1) @ inv(B1) - np.transpose(M2) @ inv(B2)))[0, 0]
            c_y = (2 * (np.transpose(M1) @ inv(B1) - np.transpose(M2) @ inv(B2)))[0, 1]
            a = (inv(B2) - inv(B1))[0, 0]
            b = y * ((inv(B2) - inv(B1))[0, 1] + (inv(B2) - inv(B1))[1, 0]) + c_x
            c = ((y ** 2) * (inv(B2) - inv(B1))[1, 1] + y * c_y
                 + abs(np.log(det(B2) / det(B1)) + 2 * np.log(P1 / P2) - np.transpose(M1) @ inv(B1) @ M1
                       + np.transpose(M2) @ inv(B2) @ M2))
            if (b ** 2 - 4 * a * c) >= 0:
                x1 = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                x2 = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if x1 != x2:
                    X1 += [[x1, y], [x2, y]]
                elif x1 == x2:
                    X1 += [x1, y]
        return np.array(X1)


    def probability_of_erroneous_classification(X1, M1, M2, B1, B2, P1, P2):  # вероятность ошибочной классификации
        X1 = np.transpose(X1).reshape(200, 2, 1)
        count = 0
        for x in X1:
            d1 = math.log(P1) - math.log(np.sqrt(det(B1))) - 0.5 * np.transpose(x - M1) @ inv(B1) @ (x - M1)
            d2 = math.log(P2) - math.log(np.sqrt(det(B2))) - 0.5 * np.transpose(x - M2) @ inv(B2) @ (x - M2)
            if d2 > d1:
                count += 1

        return count / len(X1)


    def task_three(X1, X2, X3):

        M1 = np.array([[1], [0]])
        M2 = np.array([[-1], [1]])
        M3 = np.array([[1], [-2]])
        B1 = np.array([[0.5, 0.07], [0.07, 0.73]])
        B2 = np.array([[0.1156, 0.0646], [0.0646, 0.5986]])
        B3 = np.array([[0.0841, 0.1218], [0.1218, 0.6253]])
        X = np.linspace(-2, 2, 100)

        R1 = byesian_boundary_diff(X, M1, M2, B1, B2, 0.5, 0.5)
        R2 = byesian_boundary_diff(X, M3, M2, B3, B2, 0.5, 0.5)
        R3 = byesian_boundary_diff(X, M1, M3, B1, B3, 0.5, 0.5)

        fig = plt.figure(figsize=(8, 6))
        fig.add_subplot(1, 1, 1)
        plt.title('Байесовская граница')
        plt.scatter(R1[:, 0], R1[:, 1], c='Orange', linewidth=5)
        plt.scatter(R2[:, 0], R2[:, 1], c='Green', linewidth=5)
        plt.scatter(R3[:, 0], R3[:, 1], c='pink', linewidth=5)
        plt.scatter(X1[0][:], X1[1][:], c='blue')
        plt.scatter(X2[0][:], X2[1][:], c='red')
        plt.scatter(X3[0][:], X3[1][:], c='black')
        plt.xlim([-2, 3])
        plt.ylim([-2, 2])
        show()

        p = probability_of_erroneous_classification(X1, M1, M3, B1, B3, 0.5, 0.5)
        print("Вероятность ошибочной классификации: ", float(p))
        print("Относительная погрешность: ", math.sqrt((1 - float(p)) / (200 * float(p))))
        print("Объем обуч выборки: ", int((1 - float(p)) / (0.05 ** 2 * float(p))))


    if __name__ == '__main__':
        Y1 = np.loadtxt("Y1.csv", delimiter='|')
        Y2 = np.loadtxt("Y2.csv", delimiter='|')
        X1 = np.loadtxt("X1.csv", delimiter='|')
        X2 = np.loadtxt("X2.csv", delimiter='|')
        X3 = np.loadtxt("X3.csv", delimiter='|')

        task_one(Y1, Y2)
        task_two(Y1, Y2)
        task_three(X1, X2, X3)