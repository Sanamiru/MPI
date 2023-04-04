from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# инициализация MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# размерность матриц
n = 500

# список чисел процессоров
sizes = [1, 2, 4, 8, 16, 32]

# список времен выполнения
times = []

# запуск программы для разных чисел процессоров
for s in sizes:
    if rank < s:
        comm_size = s
        comm_rank = rank

        # количество строк в каждом блоке
        m = int(n / comm_size)

        # создание блоков матрицы A и B на каждом процессоре
        A = np.random.rand(m, n)
        B = np.random.rand(n, m)
        C = np.zeros((m, m))

        # установка начального времени
        if rank == 0:
            start_time = MPI.Wtime()

        # передача блоков B на все процессоры
        B = comm.bcast(B, root=0)

        # перемножение блоков матрицы A и B
        for i in range(m):
            for j in range(m):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]

        # сбор результатов с всех процессоров на процессоре 0
        C_final = None
        if rank == 0:
            C_final = np.zeros((n, n))
        comm.Gather(C, C_final[m * rank:m * (rank + 1)], root=0)

        # вывод времени выполнения
        if rank == 0:
            end_time = MPI.Wtime()
            times.append(end_time - start_time)

# построение графика
if rank == 0:
    plt.plot(sizes, times, 'o-')
    plt.xlabel('Количество процессоров')
    plt.ylabel('Время выполнения (секунды)')
    plt.title(f"Размерность матрицы = {n}")
    plt.savefig(f"plot.png")
    plt.show()