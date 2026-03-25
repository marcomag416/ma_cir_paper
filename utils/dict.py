import os
import csv


def prepend_key_to_dict(key_prefix: str, d: dict) -> dict:
    """Prepends a key prefix to all keys in a dictionary.

    Args:
        key_prefix (str): The prefix to prepend.
        d (dict): The original dictionary.

    Returns:
        dict: A new dictionary with the key prefix prepended to all keys.
    """
    return {f"{key_prefix}{k}": v for k, v in d.items()}

def save_to_csv(data: dict, file_path: str):
    """Saves a dictionary to a CSV file.

    Args:
        data (dict): The data to save.
        file_path (str): The path to the CSV file.
    """

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    if os.path.exists(file_path):
        print(Warning(f"File {file_path} already exists. Overwriting."))

    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Metric', 'Value'])
        for key, value in data.items():
            writer.writerow([key, value])

def save_or_update_csv(data: dict, file_path: str):
    """Saves a dictionary to a CSV file, or updates it if it already exists.

    Args:
        data (dict): The data to save or update.
        file_path (str): The path to the CSV file.
    """
    if os.path.exists(file_path):
        # Read existing data
        existing_data = {}
        with open(file_path, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip header
            for row in reader:
                existing_data[row[0]] = row[1]

        # Update existing data with new data
        existing_data.update(data)

        # Write updated data back to CSV
        with open(file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Metric', 'Value'])
            for key, value in existing_data.items():
                writer.writerow([key, value])
    else:
        save_to_csv(data, file_path)