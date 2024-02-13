# -*- encoding: utf-8 -*-
"""
Copyright (c) 2024 - present Airgas
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px

from processdata.load import loadData
# date,term,sku,position,clicks

# Click-Through Rate (CTR) Analysis
def ctr_Analysis():
    try:
        print("I am inside ctr_Analysis ")
        df = loadData()
        print(df.head())
        # Analyze top searched keywords
        # Calculate Click-Through Rate (CTR)
        df['CTR'] = df['clicks'] / df.groupby('sku')[
            'clicks'].transform('sum') * 100

        # Calculate Average CTR for each keyword
        avg_ctr = df.groupby('term')['CTR'].mean().reset_index()

        # Create a bar chart
        fig = px.bar(avg_ctr, x='term', y='CTR', labels={'x': 'Keyword', 'y': 'Click-Through Rate (%)'},
                     title='Average Click-Through Rate by Keyword')
        plot_div = plot(fig, output_type='div', include_plotlyjs=True)
        return plot_div
    except FileNotFoundError:
        print("File not found! Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Search Result Position Analysis:
def search_PositionAnalysis():
    try:
        df = loadData()
        print(df.head())
        # Extracting page number and position from 'Product Position'
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
        # Calculate the average position of each keyword in search results
        avg_position = df.groupby('term')['rank'].mean().reset_index()

        # Create a bar chart
        fig = px.bar(avg_position, x='term', y='rank', labels={'x': 'Keyword', 'y': 'Average Position'},
                     title='Average Search Result Position by Keyword')
        plot_div = plot(fig, output_type='div', include_plotlyjs=True)
        return plot_div
    except FileNotFoundError:
        print("File not found! Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Rank vs. Clicks Analysis:
def ranksVSclicks():
    try:
        df = loadData()
        print(df.head())
        # Extracting page number and position from 'Product Position'
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
        # Calculate the average position of each keyword in search results
        # Create a scatter plot of Rank vs. Clicks
        fig = px.scatter(df, x='rank', y='clicks', color='term',
                         title='Rank vs. Clicks Analysis', labels={'Rank of the product': 'Rank',
                                                                  'Total Clicks for that sku': 'Clicks'})
        plot_div = plot(fig, output_type='div', include_plotlyjs=True)
        return plot_div
    except FileNotFoundError:
        print("File not found! Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

def top_SearchKeywords():
    try:
        df = loadData()
        # print(df.head())
        # Analyze top searched keywords
        top_keywords = df['term'].value_counts().head(10)
        # Create a bar chart
        fig = px.bar(top_keywords, x=top_keywords.index, y=top_keywords.values, labels={'x': 'Keyword', 'y': 'Search Volume'},
                     title='Top Searched Keywords')

        # Convert Plotly figure to JSON for rendering in Flask
        # plot_json = fig.to_json()
        plot_div = plot(fig, output_type='div', include_plotlyjs=True)
        return plot_div
    except FileNotFoundError:
        print("File not found! Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")