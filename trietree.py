import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os

matplotlib.rc("font", family="SimHei")

class TrieNode:
    def __init__(self, prefix="", value=None, parent=None):
        self.prefix = prefix
        self.children = {}
        self.is_end = False
        self.value = value
        self.positions = []
        self.parent = parent

class Trie:
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.low_trie = Trie._SubTrie()
        self.high_trie = Trie._SubTrie()

    def insert(self, number: int, index: int):
        if number < self.threshold:
            return self.low_trie.insert(number, index)
        else:
            return self.high_trie.insert(number, index)

    def display(self):
        print("🔽 小于门限的子树:")
        self.low_trie.display()
        print("\n🔼 大于等于门限的子树:")
        self.high_trie.display()

    class _SubTrie:
        def __init__(self):
            self.root = TrieNode()

        def insert(self, number: int, index: int):
            num_str = str(number)
            node = self.root
            matched_value = None

            while num_str:
                for key in node.children:
                    common_prefix = self._common_prefix(num_str, key)
                    if common_prefix:
                        if common_prefix == key:
                            node = node.children[key]
                            num_str = num_str[len(common_prefix):]
                            break
                        else:
                            old_child = node.children.pop(key)
                            new_node = TrieNode(common_prefix, parent=node)
                            node.children[common_prefix] = new_node

                            new_suffix = key[len(common_prefix):]
                            old_child.prefix = new_suffix
                            old_child.parent = new_node
                            new_node.children[new_suffix] = old_child

                            node = new_node
                            num_str = num_str[len(common_prefix):]
                            break
                else:
                    for key, child in node.children.items():
                        if child.is_end and self._is_within_2_percent(number, child.value):
                            matched_value = child.value
                            child.positions.append(index)
                            return matched_value

                    new_node = TrieNode(num_str, number, parent=node)
                    node.children[num_str] = new_node
                    node = new_node
                    num_str = ""

            node.is_end = True
            node.value = number
            node.positions.append(index)
            return matched_value or number

        def _common_prefix(self, str1, str2):
            min_len = min(len(str1), len(str2))
            for i in range(min_len):
                if str1[i] != str2[i]:
                    return str1[:i]
            return str1[:min_len]

        def _is_within_2_percent(self, num1, num2):
            if num2 == 0:
                return num1 == 0
            return abs(num1 - num2) / abs(num2) <= 0.18

        def display(self, node=None, level=0, prefix=""):
            if node is None:
                node = self.root
                print(" (root)")

            children = list(node.children.items())
            for i, (key, child) in enumerate(children):
                is_last = (i == len(children) - 1)
                connector = "└── " if is_last else "├── "
                print(prefix + connector + key + (" (END)" if child.is_end else "") +
                      (f" [Positions: {child.positions}]" if child.is_end else ""))
                extension = "    " if is_last else "│   "
                self.display(child, level + 1, prefix + extension)

def insert_df_columns_into_trie(df, threshold):
    for column_name in df.columns:
        if column_name in ('gridid', 'head_12'):
            continue

        if df[column_name].dtype == 'object' and  'value' not in column_name:
            print(f"⚠️ 跳过字符串列: {column_name}")
            continue
        trie = Trie(threshold=threshold)

        new_column = []
        for i, value in df[column_name].items():
            if pd.isna(value):
                new_column.append(value)
            else:
                matched_value = trie.insert(int(value), i)
                # print(value, matched_value)
                if 'value' in column_name:
                    new_column.append(str(int(matched_value)))
                else :
                    new_column.append(matched_value)

        df[column_name] = new_column
        # print(df[column_name])
    
    print("✅ 所有列处理完成，原列已被替换为合并后的新列")
    # print(df)
    return df

def calculate_high_cardinality(csv_file, threshold=32):
    df = pd.read_csv(csv_file)
    cardinality = {col: df[col].nunique(dropna=False) for col in df.columns}
    high_cardinality = {col: count for col, count in cardinality.items() if count > threshold}
    return high_cardinality

if __name__ == "__main__":
    folder_path = "data"
    threshold_value = 52  # ✅ 你可以在这里设置门限值
    output_folder = "data/processed_data"
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not file_path.endswith(".csv"):
            continue

        print(f"\n📂 正在处理文件: {file_name}")
        df = pd.read_csv(file_path)

        new_df = insert_df_columns_into_trie(df, threshold=threshold_value)

        processed_file = os.path.join(output_folder, f"{file_name}_processed.csv")
        new_df.to_csv(processed_file, index=False)
        print(f"🎉 文件 {file_name} 处理完成，结果保存到: {processed_file}")
