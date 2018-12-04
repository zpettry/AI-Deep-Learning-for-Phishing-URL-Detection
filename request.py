#!/usr/bin/env python
"""
This file will make a simple request to the Flask API for URL processing.
""" 
import argparse
import requests

def main(url):

        # Define URL for Flask API endpoint.
        KERAS_REST_API_URL = "http://127.0.0.1:45000/predict"

        # Set the payload to JSON format.
        payload = {"url": url}

        # Submit the POST request.
        r = requests.post(KERAS_REST_API_URL, json=payload)
        response = r.json()

        # Ensure the request was sucessful.
        if response["success"]:
                # Loop over the predictions and display them.
                print(response['predictions'])

        # Otherwise, the request failed.
        else:
                print("Request failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rock and roll.')
    parser.add_argument(
        '-u',
        dest='url',
        action='store',
        required=True,
        help="This is the url."
    )

    args = parser.parse_args()

    main(**vars(args))


