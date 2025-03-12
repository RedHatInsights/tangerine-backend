import glob
import os

import requests

DEFAULT_URL = "http://localhost:3000/api/agents/"
DEFAULT_SOURCE = "default"


def upload_files(source, directory_path, url, agent_id, html, bearer_token):
    files = glob.glob(os.path.join(directory_path, "**", "*.rst"), recursive=True)
    files.extend(glob.glob(os.path.join(directory_path, "**", "*.md"), recursive=True))
    files.extend(glob.glob(os.path.join(directory_path, "**", "*.adoc"), recursive=True))
    if html:
        files.extend(glob.glob(os.path.join(directory_path, "**", "*.html"), recursive=True))

    total_files = len(files)
    batch_size = 10
    num_batches = (total_files + batch_size - 1) // batch_size

    headers = {}
    if bearer_token:
        print("Using bearer auth")
        headers = {"Authorization": f"Bearer {bearer_token}"}

    url = f"{url}/{agent_id}/documents"

    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_files = files[start:end]
        # skip 404.html files
        files_to_upload = [
            ("file", (file_path, open(file_path, "rb"), "text/plain"))
            for file_path in batch_files
            if not file_path.endswith("404.html")
        ]

        response = requests.post(
            url, files=files_to_upload, data={"source": source}, headers=headers
        )

        if response.status_code == 200:
            print(f"Batch {i + 1}/{num_batches} uploaded successfully.")
        else:
            print(f"Error uploading batch {i + 1}/{num_batches}: {response.text}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload .rst/.md/.html/.adoc files to a tangerine agent."
    )
    parser.add_argument("--html", help="Include html docs", default=False, action="store_true")
    parser.add_argument("--bearer-token", type=str, help="Authorization bearer token")
    parser.add_argument("--agent-id", type=int, help="agent ID of the tangerine agent.")
    parser.add_argument(
        "--source",
        type=str,
        help="Name of document source. (default: 'default')",
        default=DEFAULT_SOURCE,
    )
    parser.add_argument("--url", type=str, help="URL for agents API", default=DEFAULT_URL)
    parser.add_argument("directory_path", type=str, help="Path to the directory containing files.")

    args = parser.parse_args()

    upload_files(
        args.source, args.directory_path, args.url, args.agent_id, args.html, args.bearer_token
    )
