import os
import filecmp

def find_and_delete_duplicates(folder_path):
    # Step 1: Get a list of all files in the folder and subfolders
    files = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                files.append(file_path)

    # Step 2: Create a dictionary to group files by their sizes
    size_to_files = {}
    for file_path in files:
        file_size = os.path.getsize(file_path)
        if file_size not in size_to_files:
            size_to_files[file_size] = []
        size_to_files[file_size].append(file_path)

    # Step 3: Filter out files that have unique sizes, leaving only potential duplicates
    duplicate_files = [files for files in size_to_files.values() if len(files) > 1]

    # Step 4: Iterate through the list of files with the same size and compare their contents
    for files in duplicate_files:
        print("Files with the same size:")
        for i in range(len(files)):
            print(f"- {files[i]}")
        
        # Step 5: Compare files and delete duplicates
        for i in range(1, len(files)):
            if filecmp.cmp(files[0], files[i], shallow=False):
                print(f"Deleting duplicate file: {files[i]}")
                os.remove(files[i])

# Define the path to the folder you want to check for duplicates
folder_path = "E:"
# Call the function to find and delete duplicates
find_and_delete_duplicates(folder_path)
