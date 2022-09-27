#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:49:05 2022

@author: thibault
"""
import sys
import generate_metadata_min as gen_meta

def show_usage():
    
    print(f'Usage : {sys.argv[0]} <file_path> <metadata.json> <mask.png>')

if __name__ == '__main__':

    if len(sys.argv) != 4:
        show_usage()
        
    else:
        file_path = sys.argv[1]
        output_json = sys.argv[2]
        output_mask = sys.argv[3] 
        gen_meta.main(file_path, output_json, output_mask)    

