import os
import shutil


def delete_files_except_original_data():
    items = os.listdir('.')
    protected_dirs = ["original_data"]
    for item in items:
        if item not in protected_dirs:
            item_path = os.path.join('.', item)
            if os.path.isdir(item_path):
                if item == "compress_data":
                    csv_dir = os.path.join(item_path, "csv")
                    if os.path.exists(csv_dir):
                        for root, dirs, files in os.walk(csv_dir, topdown=False):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    os.remove(file_path)
                                except OSError as e:
                                    print(f"Error deleting file {file_path}: {e}")
                            for dir in dirs:
                                dir_path = os.path.join(root, dir)
                                try:
                                    shutil.rmtree(dir_path)
                                except OSError as e:
                                    print(f"Error deleting directory {dir_path}: {e}")
                else:
                    for root, dirs, files in os.walk(item_path, topdown=False):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.remove(file_path)
                            except OSError as e:
                                print(f"Error deleting file {file_path}: {e}")
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            try:
                                shutil.rmtree(dir_path)
                            except OSError as e:
                                print(f"Error deleting directory {dir_path}: {e}")


if __name__ == "__main__":
    delete_files_except_original_data()
    