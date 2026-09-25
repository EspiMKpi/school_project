import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_nb.py <notebook.ipynb>")
        sys.exit(1)
        
    filename = sys.argv[1]
    print(f"--> Executing {filename} using WindowsSelectorEventLoopPolicy...")
    with open(filename, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(nb, timeout=1800, kernel_name="ass4")
    client.execute()
    
    with open(filename, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print(f"--> [SUCCESS] Successfully executed and saved {filename}")
