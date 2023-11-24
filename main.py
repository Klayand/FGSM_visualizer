import subprocess
import os


def run_streamlit():
    py_process = subprocess.call(['streamlit', 'run',  os.path.abspath(os.path.join(os.getcwd(), 'app.py'))])

run_streamlit()

