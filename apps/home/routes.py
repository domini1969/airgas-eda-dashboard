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
import pysolr
import json

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
    try:
        print("I am inside plot blueprint")
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

@blueprint.route('/search_Airgas', methods=['POST'])
@login_required
def search_Airgas():
    try:
        if request.method == 'POST':
            print("I am inside search_Airgas blueprint")
            keywords = request.form.get('keywords')
            print(keywords)
            response_text = search_func(keywords)
            print(f"See here: {response_text}")
            return jsonify({'response_text': response_text})
    except Exception as ex:
        print(f"Error during site search: {ex}")
        print(response_text)
        return jsonify({'error': 'An error occurred during request processing.'}), 500

@blueprint.route('/harvest_Airgas', methods=['POST'])
@login_required
def harvest_Airgas():
    try:
        if request.method == 'POST':
            print("I am inside optimize_Airgas blueprint")
            keywords = request.form.get('keywords')
            print(keywords)
            response_text = harvest_func(keywords)
            print(f"See here: {response_text}")
            return jsonify({'response_text': response_text})
    except Exception as ex:
        print(f"Error during Optimize: {ex}")
        print(response_text)
        return jsonify({'error': 'An error occurred during request processing.'}), 500


@blueprint.route('/newSearch_Airgas', methods=['POST'])
@login_required
def newSearch_Airgas():
    solr_url = "https://airgas:airgas@ss448435-rvqwqbon-us-east-1-aws.searchstax.com/solr/"
    collection = "product2"
    try:
        if request.method == 'POST':
            print("I am inside newSearch_Airgas blueprint")
            keyword = request.form.get('keyword')
            print(keyword)
            # Call the get_search_results function
            data = exec_search(solr_url, collection, keyword)
            # Check if search_results is not None
            if data is not None:
                # Return the search results as JSON
                return jsonify({"search_results": data})
            else:
                # Return the error message as JSON with appropriate status code for error
                return jsonify({"error": "An error occurred while executing the search."}), 500
    except Exception as ex:
        print(f"Error during request processing: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

@blueprint.route('/delete_Airgas', methods=['POST'])
@login_required
def delete_Airgas():
    try:
        if request.method == 'POST':
            print("I am inside delete_Airgas blueprint")
            keywords = request.form.get('keywords')
            print(keywords)
            # Deleting all documents before indexing new data
            solr_url = "https://airgas:airgas@ss448435-rvqwqbon-us-east-1-aws.searchstax.com/solr/product2"
            success, message = delete_func(solr_url)
            if success:
                print("All documents deleted successfully.")
            else:
                print("Failed to delete documents:", message)
                return jsonify({'error': message}), 500
            # Return success response
            return jsonify({'success': 'All documents deleted successfully.'})
    except Exception as ex:
        print(f"Error during request processing: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

@blueprint.route('/synonyms_Airgas', methods=['POST'])
@login_required
def synonyms_Airgas():
    try:
        if request.method == 'POST':
            print("I am inside synonyms_Airgas blueprint")
            keywords = request.form.get('keywords')
            print(keywords)
            solr_url = "https://airgas:airgas@ss448435-rvqwqbon-us-east-1-aws.searchstax.com/solr/product2"
            # Define payload to enable configEdit
            synonyms = {
                "laptop, notebook",
                "keyboard, keypad"
            }
            response_text = synonyms_func(solr_url, synonyms)
            print(f"See here: {response_text}")
            return jsonify({'response_text': response_text})
    except Exception as ex:
        print(f"Error during adding synonyms: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

@blueprint.route('/config_Airgas', methods=['POST'])
@login_required
def config_Airgas():
    try:
        if request.method == 'POST':
            print("I am inside config_Airgas blueprint")
            keywords = request.form.get('keywords')
            print(keywords)
            solr_url = "https://airgas:airgas@ss448435-rvqwqbon-us-east-1-aws.searchstax.com/solr/product"
            # Define payload to enable configEdit
            payload = {
                "add-updateprocessor": {
                    "name": "configEdit",
                    "class": "solr.ConfigUpdateProcessorFactory",
                    "disable.configEdit": "false"
                }
            }
            response_text = config_func(solr_url, payload)
            print(f"See here: {response_text}")
            return jsonify({'response_text': response_text})
    except Exception as ex:
        print(f"Error during solr config changes: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

def search_func(keywords):
    # List to store the processed product data
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
                    'Position': position_counter,  # Updated position calculation
                    'partNumber': product_code,
                    'shortDescription_text_en': product_name
                }
                keyword_results.append(product_data_item)  # Append product data to keyword results
        # Append keyword results to product data
        product_data.append({'keyword': keyword, 'results': keyword_results})
    return product_data

def harvest_func(keywords):
    # Initialize Solr connection
    solr = pysolr.Solr('https://airgas:airgas@ss448435-rvqwqbon-us-east-1-aws.searchstax.com/solr/product2', always_commit=True)
    # List to store the processed product data
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
                    'Position': position_counter,  # Updated position calculation
                    'partNumber': product_code,
                    'shortDescription_text_en': product_name,
                    'shortDescription_text_en_splitting': product_name,
                    'shortDescription_text_en_splitting_tight': product_name,
                    'shortDescription_stem': product_name,
                    'shortDescription_ngram': product_name,
                    'shortDescription_suggest': product_name
                }
                solr.add([product_data_item])
                # Check if the product code already exists in Solr
                if solr.search(f'partNumber:{product_code}'):
                    try:
                        # Add document to Solr if it doesn't exist already
                        solr.add([product_data_item])
                    except Exception as e:
                        print(f"Error adding document to Solr: {e}")
                else:
                    print(f"Product with partNumber {product_code} already exists in Solr.")
                keyword_results.append(product_data_item)  # Append product data to keyword results
        # Append keyword results to product data
        product_data.append({'keyword': keyword, 'results': keyword_results})
    return product_data

# Solr - Fetch results
def exec_search(solr_url, collection, keyword):
    try:
        query = {
            "query": keyword,
            "fields": ["partNumber", "shortDescription_text_en", "score"],
            "limit": 60,
            "params": {
                "defType": "edismax",
                "qf": "shortDescription_text_en_splitting_tight",
                "pf": "shortDescription_text_en_splitting_tight",
                "sort": "score desc",
                # "spellcheck": "true",  # Enable spell check
                # "spellcheck.dictionary": "default",
                # "spellcheck.q": keyword  # Specify the query for spell check
            }
        }

        print(query)
        response = requests.get(f"{solr_url}{collection}/select", json=query)
        response.raise_for_status()  # Raise an exception for HTTP errors
        search_results = response.json()["response"]["docs"]
        print(search_results)
        return search_results # Return search results and no error message
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error occurred: {e}"
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection error occurred: {e}"
    except requests.exceptions.Timeout as e:
        return None, f"Timeout error occurred: {e}"
    except requests.exceptions.RequestException as e:
        return None, f"Request error occurred: {e}"
    except KeyError as e:
        return None, f"Key error occurred: {e}"
    except Exception as e:
        return None, f"An error occurred: {e}"

def delete_func(solr_url):
    try:
        # Create a Solr connection
        solr = pysolr.Solr(solr_url, always_commit=True)
        # Delete all documents from the Solr collection
        solr.delete(q='*:*')
        # Commit the changes
        solr.commit()
        return True, "All documents deleted successfully."
    except Exception as e:
        return False, f"An error occurred: {e}"

def synonyms_func(solr_url, synonyms):
    # Prepare the request URL
    url = f"{solr_url}/config"
    print(url)
    # Define the payload to update synonyms
    payload = {
        "update-requesthandler": {
            "/update": {
                "defaults": {
                    "update.chain": "add-unknown-fields-to-the-schema",
                    "autoCommitMaxTime": "15000"
                }
            }
        },
        "add-field": {
            "name": "text",
            "type": "text_general",
            "multiValued": True,
            "indexed": True,
            "stored": True
        },
        "add-field-type": {
            "name": "text_general",
            "class": "solr.TextField",
            "positionIncrementGap": "100",
            "indexAnalyzer": {
                "tokenizer": {
                    "class": "solr.StandardTokenizerFactory"
                },
                "filters": [
                    {"class": "solr.LowerCaseFilterFactory"}
                ]
            },
            "queryAnalyzer": {
                "tokenizer": {
                    "class": "solr.StandardTokenizerFactory"
                },
                "filters": [
                    {"class": "solr.LowerCaseFilterFactory"}
                ]
            }
        },
        "add-copy-field": {
            "source": "*",
            "dest": "text"
        },
        "update-synonyms": {
            "synonymMappings": synonyms
        }
    }

    # Send the request
    response = requests.post(url, json=payload)
    print(response)
    # Check if the request was successful
    if response.status_code == 200:
        print(f"Added synonyms successfully. Status code: {response.status_code}")
        return response
    else:
        print(f"Failed to add synonyms. Status code: {response.status_code}")
        return response

def config_func(solr_url, payload):
    # Define Solr core URL and configuration update endpoint
    url = f"{solr_url}/config"
    print(url)
    # Send POST request to update configuration
    response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    print(response)
    # Check if the request was successful
    if response.status_code == 200:
        print("Configuration update successful.")
    else:
        print(f"Error: Configuration update failed. Status code: {response.status_code}")
    # Extract relevant information from the response
    response_data = {
        'status_code': response.status_code,
        'text': response.text,
        # Add more attributes if needed
    }
    # Serialize the response data to JSON
    response_json = json.dumps(response_data)
    return response_json
