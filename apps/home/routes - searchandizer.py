# -*- encoding: utf-8 -*-
"""
Copyright (c) 2023 - present Commerce Exchange Ltd.
"""
from bs4 import BeautifulSoup
from apps.home import blueprint
from flask import render_template, request, session
from flask_login import login_required
from jinja2 import TemplateNotFound
from flask import jsonify
import uuid
import zlib
import json
# Plot related imports
from ltr.plots import plot_pairwise_data
from ltr.plots import plot_judgments
import matplotlib.pyplot as plt
from io import BytesIO
import base64
# solr related imports
# From the directory \apps\home\aips.py
from apps.home.aips import *
from apps.home.cxltr import *

# Use a dictionary as server-side storage
server_side_storage = {}

client = SolrClient(solr_base=SOLR_URL)
collection = "products"

@blueprint.route('/index')
@login_required
def index():
    return render_template('home/index.html', segment='index')


# Step 0:
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

# Initialize df_All and  df_GroupByQuery
df_All = None
df_GroupByQuery = None
chunk_size = 10000
@blueprint.route('/load_Sessions', methods=['POST'])
@login_required
def load_Sessions():
    try:
        global df_All  # Ensure you're using the global variable
        global df_GroupByQuery
        if request.method == 'POST':
            print("I am inside load_Sessions blueprint")
            # Retrieve action and options from the request
            action = request.form.get('action')
            options = request.form.get('options')
            print(action)
            print(options)
            # Check if df_All is already in the session
            session_identifier = session.get('df_identifier')
            if session_identifier and session_identifier in server_side_storage:
                # Retrieve df_All from the session storage
                df_All_dict = server_side_storage[session_identifier]
                df_All = pd.DataFrame(df_All_dict['data'], columns=df_All_dict['columns'])

            # If df_All is still None, load it and store it in the session
            if df_All is None:
                try:
                    df_All = load_all_sessions()
                    session_identifier = str(uuid.uuid4())
                    session['df_identifier'] = session_identifier
                    server_side_storage[session_identifier] = df_All.to_dict(orient='split')
                    print("Printing df_All head.......")
                    print(df_All.head())
                except Exception as e:
                    print(f"An error occurred while loading data: {e}")
                    return jsonify({'error': 'An error occurred during data loading.'}), 500

            # Filter df_All based on the 'query' column
            if options == 'All':
                df_GroupByQuery = df_All
            else:
                df_GroupByQuery = df_All[df_All['query'] == options]

            # Chunk the data and prepare the response
            df_Chunk = df_GroupByQuery[0:chunk_size]
            data_dict = df_Chunk.to_dict(orient='records')

            return jsonify({
                'data': data_dict,
                'recordsByQuery': len(df_GroupByQuery),
                'recordsFetched': len(df_Chunk),
                'draw': int(request.form.get('draw', 1)),
            })

    except Exception as ex:
        print(f"An error occurred: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

@blueprint.route('/create_Judgements', methods=['POST'])
@login_required
def create_Judgements():
    try:
        global df_All  # Ensure you're using the global variable
        if request.method == 'POST':
            print("I am inside create_Judgements blueprint")
            # Retrieve action and options from the request
            action = request.form.get('actionCJ')
            options = request.form.get('optionsCJ')
            print(action)
            print(options)
            # If df_All is still None, load it and store it in the session
            if df_All is None:
                try:
                    df_All = load_all_sessions()
                    session_identifier = str(uuid.uuid4())
                    session['df_identifier'] = session_identifier
                    server_side_storage[session_identifier] = df_All.to_dict(orient='split')
                    print(df_All.head())
                except Exception as e:
                    print(f"An error occurred while loading data: {e}")
                    return jsonify({'error': 'An error occurred during data loading.'}), 500

            if options == 'All':
                simulated_queries = ['dryer', 'bluray', 'blue ray', 'headphones', 'ipad', 'iphone', 'kindle', 'lcd tv', 'macbook', 'nook', 'star trek', 'star wars', 'transformers dark of the moon']
            else:
                simulated_queries = [options]

            judgments = []
            for qid, query in enumerate(simulated_queries):
                sdbn = sessions_to_sdbn(df_All, query)
                judgments.extend(sdbn_to_judgments(sdbn, query, qid))
            print(judgments)
            #print(sdbn)
            sdbn = sdbn.reset_index()
            # Convert sdbn to a serializable format (list of dictionaries)
            serialized_sdbn = sdbn.to_dict(orient='records')
            print(serialized_sdbn)
            # Convert Judgments to a serializable format (list of dictionaries)
            serialized_judgements = []
            for judgment in judgments:
                judgment_dict = {
                    'grade': judgment.grade,
                    'qid': judgment.qid,
                    'keywords': judgment.keywords,
                    'doc_id': judgment.doc_id,
                    'features': judgment.features,
                    'weight': judgment.weight,
                }
                serialized_judgements.append(judgment_dict)
            # Return both sdbn and judgements in the result as JSON
            return jsonify({'sdbn': serialized_sdbn, 'judgements': serialized_judgements})
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

@blueprint.route('/logNnorm_Features', methods=['POST'])
@login_required
def logNnorm_Features():
    try:
        if request.method == 'POST':
            print("I am inside logNnorm_Features blueprint")
            # Retrieve action and options from the request
            action = request.form.get('actionCJ')
            options = request.form.get('optionsCJ')
            print(action)
            print(options)
            #response = create_n_load_features_to_solr()
            #print(response)
            logged_judgments = exec_logFeatures()
            # Convert Logged Judgments to a serializable format (list of dictionaries)
            serialized_logged_judgments = []
            for logged_judgment in logged_judgments:
                logged_judgment_dict = {
                    'grade': logged_judgment.grade,
                    'qid': logged_judgment.qid,
                    'keywords': logged_judgment.keywords,
                    'doc_id': logged_judgment.doc_id,
                    'features': logged_judgment.features,
                    'weight': logged_judgment.weight,
                }
                serialized_logged_judgments.append(logged_judgment_dict)
            print("logged_judgments------------------------------------------------------")
            print(logged_judgments)
            for logged_judgment in logged_judgments:
                print(f"Logged Judgment at {logged_judgment}:")
                for key, value in logged_judgment.__dict__.items():
                    print(f"{key}: {value}")
                print("\n")
            print("\n")
            means, std_devs, normed_judgments = normalize_features(logged_judgments)
            # Convert Logged Judgments to a serializable format (list of dictionaries)
            serialized_normed_judgments = []
            for normed_judgment in normed_judgments:
                normed_judgments_dict = {
                    'grade': normed_judgment.grade,
                    'qid': normed_judgment.qid,
                    'keywords': normed_judgment.keywords,
                    'doc_id': normed_judgment.doc_id,
                    'features': normed_judgment.features,
                    'weight': normed_judgment.weight,
                }
                serialized_normed_judgments.append(normed_judgments_dict)
            print("normed_judgments------------------------------------------------------")
            for normed_judgment in normed_judgments:
                print(f"Normed Judgment at {normed_judgment}:")
                for key, value in normed_judgment.__dict__.items():
                    print(f"{key}: {value}")
                print("\n")
            print("\n")
            print("means: ", means)
            print("\n")
            print("std_devs: ", std_devs)
            print("\n")
            # Use the Agg backend
            plt.switch_backend('Agg')
            # Plot the pairwise data
            plt.figure()
            # Assuming plot_pairwise_data takes care of plotting features and predictors with labels
            # Blue Ray, BluRay, Dryer, Nook.
            plot_judgments(qids=[0, 1, 2, 9],
                           xlabel="name BM25 Std Devs",
                           ylabel="shortDescription BM25 Std Devs",
                           title_prepend="Normalized features for queries:",
                           judg_list=normed_judgments)
            # Add legends to the plot
            legend_labels = ['Blue Ray', 'BluRay', 'Dryer', 'Nook']
            plt.legend(legend_labels, loc='upper left')  # You can adjust the 'loc' parameter for the legend position
            # Convert the plot to bytes
            image_NormFeatures = BytesIO()
            plt.savefig(image_NormFeatures, format='png')
            image_NormFeatures.seek(0)
            # Encode the image as base64
            encoded_image_NF = base64.b64encode(image_NormFeatures.read()).decode('utf-8')
            return jsonify({'serialized_logged_judgments': serialized_logged_judgments,'serialized_normed_judgments': serialized_normed_judgments, 'encoded_image_NF': encoded_image_NF})
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

# Step 4:
@blueprint.route('/pairwise_Transform', methods=['POST'])
@login_required
def pairwise_Transform():
    try:
        if request.method == 'POST':
            print("I am inside pairwise_Transform blueprint")
            # Retrieve action and options from the request
            action = request.form.get('actionPT')
            print(action)
            logged_judgments = exec_logFeatures()
            means, std_devs, normed_judgments = normalize_features(logged_judgments)
            feature_deltas, predictor_deltas = pairwise_transform(normed_judgments)
            # Convert NumPy arrays to lists
            feature_deltas_list = feature_deltas.tolist() if isinstance(feature_deltas, np.ndarray) else feature_deltas
            predictor_deltas_list = predictor_deltas.tolist() if isinstance(predictor_deltas, np.ndarray) else predictor_deltas
            # Store feature_deltas and predictor_deltas in session
            session['means'] = means
            session['std_devs'] = std_devs
            session['feature_deltas'] = feature_deltas_list
            session['predictor_deltas'] = predictor_deltas_list
            #print(feature_deltas)
            #print(predictor_deltas)
            # Filter down to a judgment list of your favorite queries out of the normalized data.
            # Blue Ray, BluRay, Dryer, Nook.
            just_these_queries = []
            for j in normed_judgments:
                if j.qid == 0 or j.qid == 1 or j.qid == 2 or j.qid == 9:
                    just_these_queries.append(j)
            # Pairwise transform just these queries
            features, predictors = pairwise_transform(just_these_queries)
            # Use the Agg backend
            plt.switch_backend('Agg')
            # Plot the pairwise data
            plt.figure()
            # Assuming plot_pairwise_data takes care of plotting features and predictors with labels
            plot_pairwise_data(features, predictors,
                               xlabel="name BM25 (Delta Std Devs)",
                               ylabel="shortDescription BM25 (Delta Std Devs)",
                               title="Pairwise Differences, Dryer, Blue Ray, Blu Ray, Nook")

            # Add legend with labels
            plt.legend(labels=['Feature[Irrelevent-Relevant]', 'Predictor[Relevant-Irrelevent]'])

            # Convert the plot to bytes
            image_streamFP = BytesIO()
            plt.savefig(image_streamFP, format='png')
            image_streamFP.seek(0)
            # Encode the image as base64
            encoded_image_FP = base64.b64encode(image_streamFP.read()).decode('utf-8')

            plot_pairwise_data(feature_deltas, predictor_deltas,
                               xlabel="name BM25 (Delta Std Devs)",
                               ylabel="shortDescription BM25 (Delta Std Devs)",
                               title="Pairwise Differences, Dryer, Blue Ray, Blu Ray, Nook")

            # Add legend with labels
            plt.legend(labels=['Feature Delta[Irrelevent-Relevant]', 'Predictor Delta[Relevant-Irrelevent]'])

            # Convert the plot to bytes
            image_streamFDPD = BytesIO()
            plt.savefig(image_streamFDPD, format='png')
            image_streamFDPD.seek(0)
            # Encode the image as base64
            encoded_image_FDPD = base64.b64encode(image_streamFDPD.read()).decode('utf-8')
            plt.close()
            # Pass the encoded image to the Jinja template
            return jsonify({'encoded_image_FP': encoded_image_FP,'encoded_image_FDPD': encoded_image_FDPD })
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

# Step 5:
@blueprint.route('/train_Model', methods=['POST'])
@login_required
def train_Model():
    try:
        if request.method == 'POST':
            print("I am inside train_Model blueprint")
            # Retrieve action and options from the request
            action = request.form.get('action')
            print(action)
            # Retrieve feature_deltas and predictor_deltas from session
            feature_deltas = session.get('feature_deltas')
            predictor_deltas = session.get('predictor_deltas')
            #print(feature_deltas)
            #print(predictor_deltas)
            if feature_deltas is None or predictor_deltas is None:
                print("feature_deltas or predictor_deltas is/are none")
                # return jsonify({'error': 'Feature deltas or predictor deltas not found in session.'}), 400

            # Convert Python lists back to NumPy arrays if needed
            feature_deltas = np.array(feature_deltas) if isinstance(feature_deltas, list) else feature_deltas
            predictor_deltas = np.array(predictor_deltas) if isinstance(predictor_deltas, list) else predictor_deltas

            # Fit SVM model
            svm_model = fit_svm_model(feature_deltas, predictor_deltas)
            session['svm_model'] = svm_model
            print("-----------Start---------------")
            print(svm_model)
            # Convert NumPy array to list for JSON serialization
            print("-----------Next----------")
            svm_model_list = svm_model.tolist() if isinstance(svm_model, np.ndarray) else svm_model
            print(svm_model_list)
            print("-----------End----------")
            print("------------ svm_model -------------------")
            return jsonify({'svm_model': svm_model_list})
    except Exception as ex:
        print(f"An error occurred: {ex}")
        return jsonify({'error': 'An error occurred during request processing.'}), 500

# Step 6:
@blueprint.route('/upload_Model', methods=['POST'])
@login_required
def upload_Model():
    try:
        if request.method == 'POST':
            print("I am inside upload_Model blueprint")
            # Retrieve action and options from the request
            action = request.form.get('action')
            print(action)
            feature_set = [
                {
                    "name": "name_bm25",
                    "store": "test",
                    "class": "org.apache.solr.ltr.feature.SolrFeature",
                    "params": {  # q=title:({$keywords})
                        "q": "name:(${keywords})"
                    }
                },
                {
                    "name": "name_constant",
                    "store": "test",
                    "class": "org.apache.solr.ltr.feature.SolrFeature",
                    "params": {  # q=title:({$keywords})
                        "q": "name:(${keywords})^=1"
                    }
                }
            ]
            means = session.get('means')
            std_devs = session.get('std_devs')
            svm_model = session.get('svm_model')
            print(collection)
            response_text = upload_linear_model_v1(collection, feature_set, svm_model, means, std_devs)
            print(f"Model upload successful. Response: {response_text}")
            return jsonify({'response_text': response_text})
    except Exception as ex:
        print(f"Error during model upload: {ex}")
        print(response_text)
        return jsonify({'error': 'An error occurred during request processing.'}), 500

@blueprint.route('/<template>', methods=['GET', 'POST'])
@login_required
def route_template(template):
    try:
        if not template.endswith('.html'):
            template += '.html'
        # Detect the current page
        segment = get_segment(request)
        context = {"segment": segment}
        if (segment == 'sb-demo.html'):
            if request.method == 'POST':
                options = request.form['options']
                searchterm = request.form['keyword']
                #print(request)
                #print(template)
                #print(segment)
                #print(options)
                #print(searchterm)
                data = exec_search(options, searchterm)
                #print(data)
                return render_template("home/" + template, segment=segment, data=data, options=options, keyword=searchterm)
            else:
                return render_template("home/" + template, segment=segment)
        elif (segment == 'ltr-demo.html'):
            if request.method == 'POST':
                searchterm = request.form['keyword']
                #data = exec_ltr_search(options, searchterm)
                dataTab1 = exec_ltr_search(options="search", searchterm=searchterm)
                dataTab2 = exec_ltr_search(options="searchltr", searchterm=searchterm)
                # print(data)
                return render_template("home/" + template, segment=segment, dataTab1=dataTab1, dataTab2=dataTab2, keyword=searchterm)
            else:
                return render_template("home/" + template, segment=segment)
        elif (segment == 'eda-tabs.html'):
            if request.method == 'POST':
                if 'add_ltr_plugin_submit' in request.form:
                    response = enable_ltr(collection)
                    print(response)
                    context["data"] = response
                elif 'create_feature_set_submit' in request.form:
                    response = create_n_load_features_to_solr()
                    print(response)
                    context["data"] = response
                elif 'log_features_submit' in request.form:
                    sessions_data = load_all_sessions()
                    print("sessions_data------------------------------------------------------")
                    print(sessions_data.head())
                    simulated_queries = ['dryer', 'bluray', 'blue ray', 'headphones', 'ipad', 'iphone',
                                        'kindle', 'lcd tv', 'macbook', 'nook', 'star trek', 'star wars',
                                         'transformers dark of the moon']
                    judgments = []
                    for qid, query in enumerate(simulated_queries):
                        sdbn = sessions_to_sdbn(sessions_data, query)
                        judgments.extend(sdbn_to_judgments(sdbn, query, qid))
                    print("sdbn------------------------------------------------------")
                    print(sdbn)
                    print(judgments)
                    logged_judgments = exec_logFeatures()
                    print("logged_judgments------------------------------------------------------")
                    print(logged_judgments)
                    for logged_judgment in logged_judgments:
                        print(f"Logged Judgment at {logged_judgment}:")
                        for key, value in logged_judgment.__dict__.items():
                            print(f"{key}: {value}")
                        print("\n")
                    print("\n")
                    means, std_devs, normed_judgments = normalize_features(logged_judgments)
                    print("normed_judgments------------------------------------------------------")
                    for normed_judgment in normed_judgments:
                        print(f"Normed Judgment at {normed_judgment}:")
                        for key, value in normed_judgment.__dict__.items():
                            print(f"{key}: {value}")
                        print("\n")
                    print("\n")
                    print("means: ", means)
                    print("\n")
                    print("std_devs: ", std_devs)
                    print("\n")
                    feature_deltas, predictor_deltas = pairwise_transform(normed_judgments)
                    print(feature_deltas)
                    print(predictor_deltas)
                    # Fit SVM model
                    svm_model = fit_svm_model(feature_deltas, predictor_deltas)
                    print("svm_model ------------------------------------------------------")
                    print(svm_model)
                    print("\n")
                    print("\n")
                    #feature_set = [...]  # Your feature set
                    #model = ...  # Your trained model
                    #means = [...]  # Means for normalization
                    #std_devs = [...]  # Standard deviations for normalization
                    feature_set = [
                        {
                            "name": "name_bm25",
                            "store": "test",
                            "class": "org.apache.solr.ltr.feature.SolrFeature",
                            "params": {  # q=title:({$keywords})
                                "q": "name:(${keywords})"
                            }
                        },
                        {
                            "name": "name_constant",
                            "store": "test",
                            "class": "org.apache.solr.ltr.feature.SolrFeature",
                            "params": {  # q=title:({$keywords})
                                "q": "name:(${keywords})^=1"
                            }
                        }
                    ]
                    try:
                        response_text = upload_linear_model_v1(collection, feature_set, svm_model, means, std_devs)
                        print(f"Model upload successful. Response: {response_text}")
                    except Exception as e:
                        print(f"Error during model upload: {e}")
                    print("Final ------------------------------------------------------")
                    print(response_text)
                    context["data"] = None
                else:
                    # Handle other form submissions or provide a default action
                    temp = None
                    # Render the template with the conditional context
                return render_template("home/" + template, **context)
            else:
                return render_template("home/" + template, segment=segment)
        else:
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

# Solr - Fetch results
def exec_search(options, searchterm):
        if options == 'search':
            collection = "products"
            #"fields": ["upc", "name", "manufacturer", "image", "score"],
            query = {
                "query": searchterm,
                "fields": ["upc", "name", "manufacturer", "score"],
                "limit": 20,
                "params": {
                    "defType": "edismax",
                    #"qf": "name, longDescription",
                    #"pf": "name, longDescription",
                    "qf": "name",
                    "pf": "name",
                    #"sort": "score desc, upc asc"
                    "sort": "score desc"
                }
            }
            search_results = None  # Initialize search_results to None before the try block
            try:
                response = requests.get(f"{SOLR_URL}/{collection}/select", json=query)
                response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx status codes)
                search_results = response.json()["response"]["docs"]
            except requests.exceptions.RequestException as e:
                # Handle request exceptions, such as network errors, timeout, etc.
                print(f"Request failed: {e}")
            except requests.exceptions.HTTPError as e:
                # Handle HTTP errors (4xx or 5xx status codes)
                print(f"HTTP error: {e}")
            except json.decoder.JSONDecodeError as e:
                # Handle JSON decoding error
                print(f"JSON decoding error: {e}")
                # You might want to set search_results to a default value or handle this situation appropriately
            except KeyError as e:
                # Handle KeyError if the expected keys are not present in the JSON response
                print(f"KeyError: {e}. Check the structure of the JSON response.")
            except Exception as e:
                # Handle other exceptions that might occur
                print(f"An unexpected error occurred: {e}")
                # You might want to set search_results to a default value or handle this situation appropriately
            # Place the return statement outside the try-except blocks
            return search_results

        elif options == 'searchsb':
            signals_boosting_collection = "signals_boosting"
            # insert into signals_boosting(query, doc, boost)
            # select q.target as query, c.target as doc, count(c.target) as boost
            # from signals c left join signals q on c.query_id = q.query_id
            # where c.type = 'click'
            # AND q.type = 'query'
            # group by q.target, c.target
            # order by count(c.target) desc
            signals_boosts_query = {
                "query": searchterm,
                "fields": ["doc", "boost"],
                "limit": 10,
                "params": {
                    "defType": "edismax",
                    "qf": "query",
                    # "mm": "2",
                    "sort": "boost desc"
                }
            }

            signals_boosts = \
                requests.get(f"{SOLR_URL}/{signals_boosting_collection}/select", json=signals_boosts_query).json()["response"]["docs"]
            print(f"Boost Documents: \n{signals_boosts}")

            product_boosts = ""
            for entry in signals_boosts:
                if len(product_boosts) > 0:  product_boosts += " "
                product_boosts += f'"{entry["doc"]}"^{str(entry["boost"])}'

            if len(product_boosts) > 0:
                print(f"\nThere is a Boost Query: \n{product_boosts}")
                signal = "yes"
            else:
                print(f"\n****** There is NO Boost Query *****: \n{product_boosts}")
                signal = "no"

            collection = "products"
            if len(product_boosts) > 0:
                query = {
                    "query": searchterm,
                    "fields": ["upc", "name", "manufacturer", "image", "score"],
                    "limit": 20,
                    "params": {
                        "defType": "edismax",
                        "indent": "true",
                        #"qf": "name, longDescription",
                        #"pf": "name, longDescription",
                        "qf": "name",
                        "pf": "name",
                        #"sort": "score desc, upc asc"
                        "sort": "score desc",
                        "boost": "sum(1,query({! df=upc v=$signals_boosting}))",
                        "signals_boosting": product_boosts
                    }
                }
            else:
                query = {
                    "query": searchterm,
                    "fields": ["upc", "name", "manufacturer", "image", "score"],
                    "limit": 20,
                    "params": {
                        "defType": "edismax",
                        "indent": "true",
                        #"qf": "name, longDescription",
                        #"pf": "name, longDescription",
                        "qf": "name",
                        "pf": "name",
                        #"sort": "score desc, upc asc"
                        "sort": "score desc"
                    }
                }
            #print(query)
            search_results = requests.get(f"{SOLR_URL}/{collection}/select", json=query).json()["response"]["docs"]
            return search_results

def exec_ltr_search(options, searchterm):
    print(options)
    print(searchterm)
    if options == 'search':
        collection = "products"
        query = {
            "query": searchterm,
            "fields": ["upc", "name", "manufacturer", "score"],
            "limit": 100,
            "params": {
                "defType": "edismax",
                # "qf": "name, longDescription",
                # "pf": "name, longDescription",
                "qf": "name",
                "pf": "name",
                # "sort": "score desc, upc asc"
                "sort": "score desc"
            }
        }
        # print({SOLR_URL})
        search_results = requests.get(f"{SOLR_URL}/{collection}/select", json=query).json()["response"]["docs"]
        return search_results
    elif options == 'searchltr':
        collection = "products"
        query = {
            "fields": ["upc", "name", "manufacturer", "score"],
            "limit": 100,
            "params": {
                "q": "{!ltr reRankDocs=60000 reRankWeight=2.0 model=test_model efi.keywords=\"" + searchterm + "\"}",
                "qf": "name upc manufacturer shortDescription longDescription",
                "defType": "edismax",
                "q": searchterm
            }
        }
        search_results = None  # Initialize search_results to None before the try block
        try:
            response = requests.get(f"{SOLR_URL}/{collection}/select", json=query)
            response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx status codes)
            search_results = response.json()["response"]["docs"]
        except requests.exceptions.RequestException as e:
            # Handle request exceptions, such as network errors, timeout, etc.
            print(f"Request failed: {e}")
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (4xx or 5xx status codes)
            print(f"HTTP error: {e}")
        except json.decoder.JSONDecodeError as e:
            # Handle JSON decoding error
            print(f"JSON decoding error: {e}")
            # You might want to set search_results to a default value or handle this situation appropriately
        except KeyError as e:
            # Handle KeyError if the expected keys are not present in the JSON response
            print(f"KeyError: {e}. Check the structure of the JSON response.")
        except Exception as e:
            # Handle other exceptions that might occur
            print(f"An unexpected error occurred: {e}")
            # You might want to set search_results to a default value or handle this situation appropriately
        # Place the return statement outside the try-except blocks
        return search_results


# Helper - Extract current page name from request
def get_airgas(keywords):
    #search_keywords = 'Disposable Particulate Respirator, disposable respirators,AR UHP300,kjgghh,MOL2300N95'
    product_data = []
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

            for idx, product in enumerate(products, start=1):
                # Get id instead of data attribute
                product_code = product['id']
                product_name_element = product.find('span', {'id': 'productName'})
                product_name = product_name_element.text.strip() if product_name_element else "N/A"

                # Create product data dictionary with position, code, and name
                product_data_item = {
                    '#Position': (page * len(products)) + idx,  # Calculate position
                    'Part#': product_code,
                    'Product Name': product_name
                }
                keyword_results.append(product_data_item)  # Append product data to keyword results

        # Append keyword results to product data
        product_data.append({'keyword': keyword, 'results': keyword_results})
    return product_data





