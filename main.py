import pandas as pd
import os
from enum import Enum
import matplotlib.pyplot as plt
from scipy.linalg import lu_solve, lu_factor

path = os.path.dirname(os.path.abspath(__file__))
pathData = os.path.join(path, "data")
pathResults = os.path.join(path, "results")
pathPlots = os.path.join(path, "plots")


class Mode(Enum):
    LAGRANGE = 1
    SPLINES = 2


MODE = Mode.LAGRANGE
# tested sets length = 512
skipped = [16, 42, 64, 85]
# challenger - 260, chelm and everest - 8
interpolation_distance_stamp = 8


def Matrix(N):
    M = []
    for _ in range(N):
        line = []
        for _ in range(N):
            line.append(0)
        M.append(line)
    return M


def splines(nodes, i, nodes_number):
    N = 4 * (nodes_number - 1)
    b = [0] * N
    A = Matrix(N)
    A[0][0] = 1
    b[0] = nodes.iloc[0][1]
    h = nodes.iloc[1][0] - nodes.iloc[0][0]
    A[1][0] = 1
    A[1][1] = h
    A[1][2] = pow(h, 2)
    A[1][3] = pow(h, 3)
    b[1] = nodes.iloc[1][1]
    A[2][2] = 1
    b[2] = 0
    h = nodes.iloc[nodes_number - 1][0] - nodes.iloc[nodes_number - 2][0]
    A[3][4 * (nodes_number - 2) + 2] = 2
    A[3][4 * (nodes_number - 2) + 3] = 6 * h
    b[3] = 0
    for k in range(1, nodes_number - 1):
        h = nodes.iloc[k][0] - nodes.iloc[k - 1][0]
        A[4 * k][4 * k] = 1
        b[4 * k] = nodes.iloc[k][1]
        A[4 * k + 1][4 * k] = 1
        A[4 * k + 1][4 * k + 1] = h
        A[4 * k + 1][4 * k + 2] = pow(h, 2)
        A[4 * k + 1][4 * k + 3] = pow(h, 3)
        b[4 * k + 1] = nodes.iloc[k + 1][1]
        A[4 * k + 2][4 * (k - 1) + 1] = 1
        A[4 * k + 2][4 * (k - 1) + 2] = 2 * h
        A[4 * k + 2][4 * (k - 1) + 3] = 3 * pow(h, 2)
        A[4 * k + 2][4 * k + 1] = -1
        b[4 * k + 2] = 0
        A[4 * k + 3][4 * (k - 1) + 2] = 2
        A[4 * k + 3][4 * (k - 1) + 3] = 6 * h
        A[4 * k + 3][4 * k + 2] = -2
        b[4 * k + 3] = 0
    x = lu_solve(lu_factor(A), b)
    altitude = 0.0
    for k in range(nodes_number - 1):
        altitude = 0.0
        if nodes.iloc[k][0] <= i <= nodes.iloc[k + 1][0]:
            for j in range(4):
                h = i - nodes.iloc[k][0]
                altitude += x[4 * k + j] * pow(h, j)
            break
    return altitude


def lagrange(nodes, i, nodes_number):
    altitude = 0.0
    for k in range(nodes_number):
        a = 1.0
        for j in range(nodes_number):
            if k != j:
                a *= (i - nodes.iloc[j][0]) / (nodes.iloc[k][0] - nodes.iloc[j][0])
        altitude += a * nodes.iloc[k][1]
    return altitude


def plots(results_i, results, name, actual_i, actual, intersects_i, intersects):
    while actual_i[-1] > intersects_i[-1]:
        actual_i.pop(-1)
        actual.pop(-1)
    plt.plot(results_i, results, linewidth=1.0)
    plt.plot(actual_i, actual, linewidth=1.0)
    plt.scatter(intersects_i, intersects)
    plt.xlabel("Distance")
    plt.ylabel("Altitude")
    plt.legend(["Interpolated profile", "Actual profile"])
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(pathPlots, name + '.png'))
    plt.show()


def interpolation(nodes, mode, filename, nodes_number):
    results_i = []
    results = []
    with open(os.path.join(pathResults, filename + ".txt"), "w") as file:
        i = nodes.iloc[0][0]
        while i <= nodes.iloc[nodes_number - 1][0]:
            interpolate = True
            for j in range(nodes_number):
                if int(nodes.iloc[j][0]) == i:
                    interpolate = False
                    result = nodes.iloc[j][1]
                    break
            if interpolate:
                if mode == Mode.LAGRANGE:
                    result = lagrange(nodes, i, nodes_number)
                elif mode == Mode.SPLINES:
                    result = splines(nodes, i, nodes_number)
            file.write(str(i) + " " + str(result) + '\n')
            results_i.append(i)
            results.append(result)
            i += interpolation_distance_stamp
    intersects_i = []
    intersects = []
    with open(os.path.join(pathResults, filename + "_nodes.txt"), "w") as file:
        for idx, row in nodes.iterrows():
            file.write(str(row['Dystans']) + " " + str(row['Wysokosc']) + '\n')
            intersects_i.append(row['Dystans'])
            intersects.append(row['Wysokosc'])
    return results_i, results, intersects_i, intersects


def txt_to_csv(filename):
    with open(os.path.join(pathData, filename), "r") as file:
        df = pd.DataFrame(columns=["Dystans", "Wysokosc"])
        for line in file.read().split('\n'):
            distance, altitude = pd.to_numeric(line.split()[0]), pd.to_numeric(line.split()[1])
            df1 = pd.DataFrame({
                "Dystans": [distance],
                "Wysokosc": [altitude]
            })
            df = pd.concat([df, df1])
        df.to_csv(os.path.join(pathData, filename + '.csv'))


def main():
    for file in os.listdir(pathData):
        if os.path.isfile(os.path.join(pathData, file)):
            df = pd.read_csv(os.path.join(pathData, file), usecols=['Dystans', 'Wysokosc'])
            for skip in skipped:
                df_nodes = pd.DataFrame(columns=["Dystans", "Wysokosc"])
                nodes_number = int(len(df) / skip)
                index_list = []
                i = 0
                j = 0
                while i < nodes_number:
                    index_list.append(j)
                    i += 1
                    j += skip
                df_nodes = pd.concat([df_nodes, df.iloc[index_list]])
                if MODE == Mode.LAGRANGE:
                    results_i, results, intersects_i, intersects = interpolation(df_nodes, Mode.LAGRANGE,
                                                                                 file.split('.')[0] + "_lagrange" + str(
                                                                                     nodes_number), nodes_number)
                    plots(results_i, results, file.split('.')[0] + "_lagrange" + str(nodes_number),
                          list(df.loc[:, "Dystans"]), list(df.loc[:, "Wysokosc"]), intersects_i, intersects)
                else:
                    results_i, results, intersects_i, intersects = interpolation(df_nodes, Mode.SPLINES,
                                                                                 file.split('.')[0] + "_splines" + str(
                                                                                     nodes_number), nodes_number)
                    plots(results_i, results, file.split('.')[0] + "_splines" + str(nodes_number),
                          list(df.loc[:, "Dystans"]), list(df.loc[:, "Wysokosc"]), intersects_i, intersects)


if __name__ == '__main__':
    main()
