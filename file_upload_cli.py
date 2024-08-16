import glob
import os

import requests


def upload_files(source, directory_path, agent_id):
    files = glob.glob(os.path.join(directory_path, "**", "*.rst"), recursive=True)
    files.extend(glob.glob(os.path.join(directory_path, "**", "*.md"), recursive=True))
    files.extend(glob.glob(os.path.join(directory_path, "**", "*.html"), recursive=True))

    total_files = len(files)
    batch_size = 10
    num_batches = (total_files + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_files = files[start:end]
        files_to_upload = [
            ("file", (file_path, open(file_path, "rb"), "text/plain")) for file_path in batch_files
        ]

        url = f"http://localhost:3000/api/agents/{agent_id}/document_upload"
        response = requests.post(url, files=files_to_upload, data={"source": source})

        if response.status_code == 200:
            print(f"Batch {i+1}/{num_batches} uploaded successfully.")
        else:
            print(f"Error uploading batch {i+1}/{num_batches}: {response.text}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload .rst/.md/.html files to a tangerine agent."
    )
    parser.add_argument("source", type=str, help="Name of document source.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing files.")
    parser.add_argument("agent_id", type=str, help="Agent ID of the tangerine agent.")

    args = parser.parse_args()

    upload_files(args.source, args.directory_path, args.agent_id)
