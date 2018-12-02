#!/usr/bin/env python
"""
This file gathers data to be used for pre-processing in training and prediction.
"""
import pandas as pd

def main():

    blacklist = 'phishing_database.csv'
    whitelist = 'whitelist.txt'

    urls = {}
    
    blacklist = pd.read_csv(blacklist)

    #Assign 0 for non-malicious and 1 as malicious for supervised learning.
    for url in blacklist['url']:
        urls[url] = 1
    
    with open(whitelist, 'r') as f:
        lines = f.read().splitlines()
        for url in lines:
            urls[url] = 0

    return urls

if __name__ == "__main__":
    main()
