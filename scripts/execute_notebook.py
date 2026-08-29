#!/usr/bin/env python3
"""Execute the master notebook in-place (verify mode) and save outputs."""
import time
import nbformat
from nbclient import NotebookClient

PATH = "/home/z/my-project/notebooks/CipherMind_Model_Training_and_Evaluation.ipynb"
t0 = time.time()
nb = nbformat.read(PATH, as_version=4)
client = NotebookClient(
    nb,
    timeout=2400,
    kernel_name="python3",
    resources={"metadata": {"path": "/home/z/my-project/notebooks"}},
)
client.execute()
nbformat.write(nb, PATH)
print(f"EXECUTED OK in {time.time() - t0:.0f}s — outputs saved to {PATH}")
