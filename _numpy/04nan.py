import numpy as np


def fill_nan_with_col_mean(t1):
    # 这段代码主要是把nan设置为列中非nan的平均值
    for i in range(t1.shape[1]):  # 遍历每一列
        temp_col = t1[:, i]
        nan_num = np.count_nonzero(temp_col != temp_col)  # 计算nan的个数
        if nan_num != 0:
            temp_not_nan = temp_col[temp_col == temp_col]  # 取出非nan的值
            temp_col[temp_col != temp_col] = temp_not_nan.mean()

    return t1


if __name__ == '__main__':
    t1 = np.arange(12).reshape((3, 4)).astype(float)
    t1[1, 2:] = np.nan
    print(t1)
    fill_nan_with_col_mean(t1)
    print(t1)
