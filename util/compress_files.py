import os
import zipfile


def create_zip_file(file_name, zip_file_name):
	with zipfile.ZipFile(zip_file_name, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
		archive.write(file_name)

for file in os.listdir('.'):
	if '.py' not in file:
		create_zip_file(file, file.replace('.txt', '.zip'))