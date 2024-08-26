import shutil
import os

def get_unique_file_path(destination_dir, filename):
    # Create a unique file path by appending a number if needed
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(destination_dir, new_filename)):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    return os.path.join(destination_dir, new_filename)

# Define source and destination directories
source_dir = r'D:\telegram\ChatExport_2024-08-21\files'
destination_dir = 'E:'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate over all files in the source directory
for filename in os.listdir(source_dir):
    # Create full file path
    file_path = os.path.join(source_dir, filename)
    
    # Check if it is a file (not a directory)
    if os.path.isfile(file_path):
        # Create a unique destination file path if needed
        dest_path = get_unique_file_path(destination_dir, filename)
        # Move file to destination directory
        shutil.move(file_path, dest_path)
        print(f'Moved: {filename} to {dest_path}')
    else:
        print(f'Skipped (not a file): {filename}')
