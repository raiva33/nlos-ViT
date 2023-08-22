import os
import datetime

def delete_files_before_date(directory_path, target_date):
    try:
        target_date = datetime.datetime.strptime(target_date, '%Y-%m-%d').date()
    except ValueError:
        print("Invalid date format. Please use the 'YYYY-MM-DD' format.")
        return

    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory path.")
        return

    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).date()

            if file_creation_time < target_date:
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    directory_path ="/home/mao/Documents/code/mitsuba2-transient-nlos/datasets/train/feature/circle"
    target_date = "2023-07-09"

    delete_files_before_date(directory_path, target_date)
