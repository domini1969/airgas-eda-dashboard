# -*- encoding: utf-8 -*-
"""
Copyright (c) 2024 - present Airgas
"""

import os
import pandas as pd
from flask import jsonify

def loadData():
    try:
        current_dir = os.path.dirname(__file__)
        print(current_dir)
        # Specify the relative path to the CSV file
        file_path = os.path.join(current_dir, "../data/LoggedIn.csv")
        df = pd.read_csv(file_path)
        # print(df.head())
        return df
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return jsonify({'error': 'An error occurred during data loading.'}), 500

