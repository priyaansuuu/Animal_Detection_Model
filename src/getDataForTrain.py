import os
import zipfile

if not os.path.exists("data"):
    os.system("kaggle datasets download -d priyaansuu/animal")

    zip_file = [f for f in os.listdir() if f.endswith(".zip")][0]

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

    os.remove(zip_file)  # cleanup zip file