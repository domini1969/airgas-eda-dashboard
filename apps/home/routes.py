# -*- encoding: utf-8 -*-
"""
Copyright (c) 2024 - present Airgas
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound

# Imports for loading and processing user click data
import os
import pandas as pd

# Plot related imports
from flask import request, jsonify
from processdata.load import loadData
from visualization.plots import *

# site search, scrapping.
import requests
from bs4 import BeautifulSoup

@blueprint.route('/index')
@login_required
def index():
    return render_template('home/index.html', segment='index')

@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:
        if not template.endswith('.html'):
            template += '.html'
        # Detect the current page
        segment = get_segment(request)
        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)
    except TemplateNotFound:
        return render_template('home/page-404.html'), 404
    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:
        segment = request.path.split('/')[-1]
        if segment == '':
            segment = 'index'
        return segment
    except:
        return None

@blueprint.route('/load_UserClick', methods=['POST'])
@login_required
def load_UserClick():
    try:
        if request.method == 'POST':
            print("I am inside load_UserClick blueprint")
            # Retrieve action and options from the request
            action = request.form.get('action')
            options = request.form.get('options')
            print(action)
            print(options)
            df = loadData()
            # Splitting page number and position
            df[['pageTemp', 'positionTemp']] = df['position'].str.split(':', expand=True)
            # Converting pageTemp and positionTemp to integers
            df['pageTemp'] = df['pageTemp'].astype(int)
            df['positionTemp'] = df['positionTemp'].astype(int)
            # Calculate rank
            df['rank'] = 0
            for i, row in df.iterrows():
                if row['pageTemp'] == 1:
                    df.at[i, 'rank'] = ((row['pageTemp'] - 1) * 20) + row['positionTemp']
                else:
                    df.at[i, 'rank'] = ((row['pageTemp']) * 20) + row['positionTemp']
            data_dict = df.to_dict(orient='records')
            return jsonify({
                'data': data_dict,
                'recordsFetched': len(df),
                'draw': int(request.form.get('draw', 1)),
            })
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500


# Reports
@blueprint.route('/plot', methods=['POST'])
@login_required
def plotMain():
    print("hahahahaha")
    try:
        print("I am inside plotMain blueprint")
        # Retrieve action and options from the request
        option = request.form.get('action')
        print(option)
        if option == 'ctr_Analysis':
            plot = ctr_Analysis()
        elif option == 'search_PositionAnalysis':
            plot = search_PositionAnalysis()
        elif option == 'ranksVSclicks':
            plot = ranksVSclicks()
        elif option == 'Top_searched_keywords':
            plot = top_SearchKeywords()
        else:
            print("No valid analysis selection")
        #print(plot)
        return jsonify({'response': plot})
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

@blueprint.route('/airGas_Search', methods=['POST'])
@login_required
def airGas_Search():
    try:
        if request.method == 'POST':
            print("I am inside get_airgas blueprint")
            keywords = request.form.get('keywords')
            print(keywords)
            response_text = get_airgas(keywords)
            #print(f"See here: {response_text}")
            return jsonify({'response_text': response_text})
    except Exception as ex:
        print(f"Error during model upload: {ex}")
        print(response_text)
        return jsonify({'error': 'An error occurred during request processing.'}), 500

# Helper - Extract current page name from request
def get_airgas(keywords):
    #search_keywords = 'Disposable Particulate Respirator, disposable respirators,AR UHP300,kjgghh,MOL2300N95'
    product_data = []
    position_counter = 0  # Initialize position counter outside the loop

    # Split search keywords by comma to create a list
    keywords_list = keywords.split(',')

    for keyword in keywords_list:
        keyword = keyword.strip()  # Remove leading/trailing whitespace
        keyword_results = []

        for page in range(3):  # Pages 0, 1, and 2
            search_url = f"https://www.airgas.com/search?q={keyword}&page={page}"
            response = requests.get(search_url)
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')

            # Use CSS selector
            products = soup.select('div.search-result-row')

            for product in products:
                # Get id instead of data attribute
                product_code = product['id']
                product_name_element = product.find('span', {'id': 'productName'})
                product_name = product_name_element.text.strip() if product_name_element else "N/A"

                # Increment position counter for each product
                position_counter += 1

                # Create product data dictionary with updated position
                product_data_item = {
                    '#Position': position_counter,  # Updated position calculation
                    'Part#': product_code,
                    'Product Name': product_name
                }
                keyword_results.append(product_data_item)  # Append product data to keyword results

        # Append keyword results to product data
        product_data.append({'keyword': keyword, 'results': keyword_results})
    return product_data