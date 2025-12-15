import numpy as np 
import osmnx as osm
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from fn_lib import *
from tqdm import tqdm

cities2extract=[
    "Delhi",
    "Mumbai",
    "Kolkata",
    "Chennai",
    "Bengaluru",
    "Hyderabad",
    "Pune",
    "Ahmedabad",
    "Jaipur",
    "Chandigarh",
    "Lucknow",
    "Kochi",
    "Bhopal",
    "Indore",
    "Nagpur",
    "Visakhapatnam"
]
city_codes = [
    "DEL",
    "MUM",
    "KOL",
    "CHE",
    "BLR",
    "HYD",
    "PUN",
    "AHM",
    "JAI",
    "CHD",
    "LKO",
    "COK",
    "BPL",
    "IND",
    "NAG",
    "VIZ"
]
#modify the following block accordingly
# if __name__ == None:
        
#     for city in tqdm(city_codes, total=len(city_codes)):
#         # print(city, city_codes[i])
#         tempG = ExtractGraph(city, code)
#         temFig = kPDF(tempG, city, save=True)
#         plt.close()
#         del tempG
#         # del tempFig