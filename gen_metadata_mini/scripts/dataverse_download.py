#!/usr/local/bin/python
# Script to download a dataset from a Dataverse (https://dataverse.org/)
#' @author
#' John Bradley: initial code
#' Thibault Tabarin: small modification
#' 


import os
import sys
import hashlib
from pyDataverse.api import NativeApi, DataAccessApi


def download_dataset(base_url, api_token, doi, directory_output):
    api = NativeApi(base_url, api_token)
    data_api = DataAccessApi(base_url, api_token)
    dataset = api.get_dataset(doi)
    files_list = dataset.json()['data']['latestVersion']['files']
    for dv_file in files_list:
        filepath = download_file(data_api, dv_file, directory_output)
        verify_checksum(dv_file, filepath)


def download_file(data_api, dv_file, directory_output):
    filepath = dv_file["dataFile"]["filename"]
    #directory_label = dv_file["directoryLabel"]
    os.makedirs(directory_output, exist_ok=True)
    filepath = os.path.join(directory_output, filepath)
    file_id = dv_file["dataFile"]["id"]
    print("Downloading file {}, id {}".format(filepath, file_id))
    response = data_api.get_datafile(file_id)
    with open(filepath, "wb") as f:
        f.write(response.content)
    return filepath


def verify_checksum(dv_file, filepath):
    checksum = dv_file["dataFile"]["checksum"]
    checksum_type = checksum["type"]
    checksum_value = checksum["value"]
    if checksum_type != "MD5":
        raise ValueError(f"Unsupported checksum type {checksum_type}")

    with open(filepath, 'rb') as infile:
        hash = hashlib.md5(infile.read()).hexdigest()
        if checksum_value == hash:
            print("Verified file checksum for {filepath}.")
        else:
            raise ValueError(f"Hash value mismatch for {filepath}: {checksum_value} vs {hash} ")


def show_usage():
   print()
   print(f"Usage: python {sys.argv[0]} <dataverse_base_url> <doi> <directory_output>\n")
   print("To specify a API token set the DATAVERSE_API_TOKEN environment variable.\n")
   print("To set the environment variable : export DATAVERSE_API_TOKEN=<my_token>")
   print()


if __name__ == '__main__':
    if len(sys.argv) != 4:
         show_usage()
         sys.exit(1)
    else:
        BASE_URL = sys.argv[1]
        DOI = sys.argv[2]
        directory_output = sys.argv[3]
        API_TOKEN = os.environ.get('DATAVERSE_API_TOKEN')
        #print(API_TOKEN)
        download_dataset(BASE_URL, API_TOKEN, DOI, directory_output)
