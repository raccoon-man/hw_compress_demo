import pandas as pd


def calculate_cardinality(series):
    """计算 Series 的基数（唯一值的数量）"""
    return series.nunique()


def left_split(df, col_name):
    """左切分函数"""
    # 对所有元素在左边补充空格，使列对齐
    max_length = df[col_name].str.len().max()
    df[col_name] = df[col_name].str.rjust(max_length)

    for i in range(1, max_length + 1):
        prefix = df[col_name].str[:i]
        if prefix.nunique() > 1:
            break

    if i == 1:
        prefix_col = pd.Series([], dtype=object)
    else:
        prefix_col = df[col_name].str[:i - 1]
    suffix_col = df[col_name].str[i - 1:]
    return prefix_col, suffix_col


def right_split(df, col_name):
    """右切分函数"""
    # 统计未切分状态下的基数
    kn_opt = calculate_cardinality(df[col_name])
    grad_kn = 1
    best_split_point = len(df[col_name].iloc[0])
    try_time = 0
    # print('初始基数：', kn_opt)
    for i in range(1, len(df[col_name].iloc[0]) + 1):
        # print('从左往右的切分点：',i)
        
        left_col = df[col_name].str[:-i]
        right_col = df[col_name].str[-i:]
        left_kn = calculate_cardinality(left_col)
        right_kn = calculate_cardinality(right_col)
        grad_kn_new = (left_kn + right_kn) / kn_opt
        # print('左基数：',left_kn, '，右基数：',right_kn)
        if grad_kn_new < grad_kn:
            # print('更新梯度')
            kn_opt = (left_kn + right_kn)
            grad_kn = grad_kn_new
            best_split_point = len(df[col_name].iloc[0]) - i
            try_time = 0
        else:
            try_time += 1
            if try_time >= 2 or i > 1:
                break
    # print('最佳切分点：',len(df[col_name].iloc[0]) -best_split_point)
    prefix_col = df[col_name].str[:best_split_point]
    if best_split_point == len(df[col_name].iloc[0]):
        suffix_col = pd.Series([], dtype=object)
    else:
        suffix_col = df[col_name].str[best_split_point:]
    # print(prefix_col, suffix_col)
    return prefix_col, suffix_col


def SegmentExecute(df, col_name):
    """主函数，对 DataFrame 进行切分"""
    
    new_dfs = []
    current_col = df
    index = 0

    # 左切分
    prefix_col, suffix_col = left_split(pd.DataFrame(current_col), col_name)
    if not prefix_col.empty:
        new_dfs.append(prefix_col)
    
    # 右切分
    prefix_col, suffix_col = right_split(pd.DataFrame(suffix_col), col_name)
    if not prefix_col.empty:
        new_dfs.append(prefix_col)
    if not suffix_col.empty:
        new_dfs.append(suffix_col)

    result_df = pd.concat(new_dfs, axis=1)
    result_df = result_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    result_df.columns = [f"{col_name}_{i}" for i in range(len(result_df.columns))]
    return result_df
    