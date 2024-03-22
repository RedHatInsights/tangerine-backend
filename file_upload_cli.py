import os
import requests
import glob

def upload_files(directory_path, agent_id):
    files = glob.glob(os.path.join(directory_path, '**', '*.rst'), recursive=True)
    files.extend(glob.glob(os.path.join(directory_path, '**', '*.md'), recursive=True))

    total_files = len(files)
    batch_size = 10
    num_batches = (total_files + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_files = files[i * batch_size: (i+1) * batch_size]
        files_to_upload = [('file', (os.path.basename(file_path), open(file_path, 'rb'), 'text/plain')) for file_path in batch_files]
        
        # for j, file_path in enumerate(batch_files):
        #     with open(file_path, 'rb') as file:
        #         files_to_upload.append(('file', (os.path.basename(file_path), file, 'text/plain')))
        #         # files_to_upload[f'file{j+1}'] = (os.path.basename(file_path), file.read())
    
        # print(files_to_upload)
        url = f'http://localhost:3000/agents/{agent_id}/document_upload'
        response = requests.post(url, files=files_to_upload)

        if response.status_code == 200:
            print(f"Batch {i+1}/{num_batches} uploaded successfully.")
        else:
            print(f"Error uploading batch {i+1}/{num_batches}: {response.text}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload .rst and .md files to a tangerine agent.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing .rst and .md files.")
    parser.add_argument("agent_id", type=str, help="Agen ID of the tangerine agent.")

    args = parser.parse_args()
    
    upload_files(args.directory_path, args.agent_id)
