import os

root_dir = os.path.dirname(__file__)
def working_dir(folder,file):
    return os.path.join(os.path.dirname(__file__), folder, file)
