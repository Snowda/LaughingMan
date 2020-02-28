"""Runs pylint on mask.py with desired attributes"""
import subprocess, sys, os, re, argparse

def main():
    """This is the main"""
    tested_file = "laugh"
    file_path = "../bin/"
    tested_file_python = file_path+tested_file+".py"

    subprocess.call("coverage run "+tested_file_python, shell=True)
    subprocess.call("coverage report", shell=True)

    apply_lint(tested_file_python)

if __name__ == '__main__':
    main()
