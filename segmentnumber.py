import pandas as pd

def str_to_int_list(series, signed):
    """尝试将字符串Series转换成整数列表，失败的用None替代"""
    int_list = []
    for s in series:
        try:
            val = int(s)
            int_list.append(val)
        except:
            int_list.append(None)
    return int_list

def is_within_32bit_range(min_val, max_val, signed):
    """判断最小最大值是否在32bit表示范围内"""
    if signed:
        return -2**31 <= min_val and max_val <= 2**31 - 1
    else:
        return 0 <= min_val and max_val <= 2**32 - 1

def get_min_max_signed(series):
    """获取Series中的最小值、最大值，以及是否有负数"""
    int_vals = str_to_int_list(series, signed=False)
    filtered = [v for v in int_vals if v is not None]
    if not filtered:
        return 0, 0, False
    min_val, max_val = min(filtered), max(filtered)
    has_negative = any(v < 0 for v in filtered)
    return min_val, max_val, has_negative

def split_column_32bit(df, col_name, prefix=""):
    """切分字段为多个子字段，每段不超过32bit表示范围"""

    col = df[col_name]
    max_len = col.str.len().max()
    col = col.str.rjust(max_len)

    i = 2
    while i <= max_len:
        left_part = col.str[:-i]
        right_part = col.str[-i:]

        # 判断右半部分是否可以32bit表示
        r_min, r_max, r_signed = get_min_max_signed(right_part)
        if not is_within_32bit_range(r_min, r_max, r_signed):
            i += 1
            continue

        # 判断左半部分是否超过32bit（若没有则停止）
        l_min, l_max, l_signed = get_min_max_signed(left_part)
        if is_within_32bit_range(l_min, l_max, l_signed):
            break

        # 后缀满足、前缀不满足，继续左移
        i += 1

        # 如果后缀刚好32bit，前缀仍超范围，退出以处理前缀
        if i > max_len:
            break

    # 切分成功，返回右侧字段，递归左侧字段
    split_index = max_len - i
    prefix_series = col.str[:split_index]
    suffix_series = col.str[split_index:]

    result = {}
    if split_index > 0:
        prefix_field = f"{prefix}{col_name}_0"
        result[prefix_field] = prefix_series

        # 检查前缀是否仍超范围，递归切分
        l_min, l_max, l_signed = get_min_max_signed(prefix_series)
        if not is_within_32bit_range(l_min, l_max, l_signed):
            nested_df = pd.DataFrame({col_name: prefix_series})
            nested_split = split_column_32bit(nested_df, col_name, prefix=prefix)
            result.pop(prefix_field)  # 删除原始未处理字段
            result.update(nested_split)

    suffix_field = f"{prefix}{col_name}_1"
    result[suffix_field] = suffix_series

    return result

def SegmentExecute(df, col_name):
    """主函数，对 DataFrame 中指定列执行多段切分"""
    df = pd.DataFrame(df.copy())
    split_result = split_column_32bit(df, col_name)

    result_df = pd.DataFrame(split_result)
    result_df = result_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return result_df
