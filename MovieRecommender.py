import numpy as np
import pandas as pd
import time as tm
import tkinter as tk
from tkinter import messagebox
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack 
from scipy.sparse import csr_matrix




# Movie recommender system
titles = []
dataFrame = pd.read_csv("TMDB_movie_dataset_v11.csv", usecols = ["title","vote_average", "vote_count", "release_date","revenue","adult","popularity","budget","original_language","runtime","genres","keywords","overview","homepage","tagline"])
dataFrame["budget"] = dataFrame["budget"].fillna(0) # missing values with 0
dataFrame["revenue"] = dataFrame["revenue"].fillna(0)  
dataFrame["ROI"] = ((dataFrame["revenue"] - dataFrame["budget"]) / dataFrame["budget"].replace(0, 1)) * 100      # Movie ROI  how well it performed (assuming movie used all its budget as initial investment) replaces 0 with 1 to avoid division by zero


#root = tk.Tk()
#root.title("Movie Recommender")
#root.geometry("900x700")  

filterDF = dataFrame[(dataFrame["vote_average"] >= 5) & (dataFrame["vote_count"] >= 50) & (dataFrame["runtime"] >= 40) & (dataFrame['original_language'] == "en")].copy()  #filter the df to remove terrible movies / non english etc



def main():
    index = 0
    print("-"*60)
    tm.sleep(1)
    print("Welcome to the Movie Recommender System - Made by Lukas.L")
    tm.sleep(1)
    print("Loading...Dataset")
    tm.sleep(1)
    print("-"*60)
    try:
        print(filterDF.head())
        print("-"*60)
        print("Loaded successfully.")
        tm.sleep(1)
    except FileNotFoundError:
        print("Error: TMDB_movie_dataset_v11.csv not found. Please download here: 'https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies'")
        return
    tm.sleep(1)
    print("Please Enter movie TITLE you like to get recommendations based on it. These movies must be in the database...")
    tm.sleep(1)
    print("-"*60)
    mov = str(input("Enter your favorite movie title or q to quit: "))
    if mov == "q":
        print("Quitting...")
        quit()
    result = filterDF [filterDF["title"].str.contains(mov,case = False, na=False)]
    print(result[["title", "tagline"]])
    if len(result) == 1:
        indexNmbr = int(input("Please Enter the Index of the movie in the results given"))
        print("Is this your movie? ")
        tm.sleep(1)
        print(filterDF.loc[indexNmbr, "title"])
        print(filterDF.loc[indexNmbr, "release_date"])
        print(filterDF.loc[indexNmbr, "vote_average"])
        print(filterDF.loc[indexNmbr, "vote_count"])  
        print(filterDF.loc[indexNmbr, "genres"])
        answer = input("y/n: ")
        if answer == "y":
            titles.append(filterDF.loc[indexNmbr, "title"])   
            index = indexNmbr 
            print("Your Fav Titles: ", titles)    

        elif answer == "n":
            print("Sorry please, try again")
        else:
            print("Invalid input")
            print("Your Fav Titles: ", titles)          
    elif len(result) > 1 and  len(result) < 50:
        while True:
            print("Multiple movies found. Please Type the Index (the number in the first left column) of the movie in the results given")
            indexNmbr = int(input("Enter your Index nmbr: "))
            resultFromID = filterDF.iloc[indexNmbr]
            print("Did you mean this?")
            tm.sleep(2)
            print(resultFromID[["title", "tagline", "genres"]])
            answer = input("y/n: or q to quit")
            if answer == "y":
                titles.append(filterDF.loc[indexNmbr, "title"])   
                index = indexNmbr
                print("Your Fav Titles: ", titles)     
                break
            elif answer == "q":
                print("Quitting...")
                quit()
            elif answer == "n":
                print("Sorry please, enter another movie or q to quit")
                continue
            else:
                print("Invalid input, input again")
                continue
    elif len(result) > 50:
        print("-"*60)
        print("Too many results found, or your answer is too vague")
        print("Please try again")
        
    elif len(result) == 0:
        print("-"*60)
        print("Movie not found in database")
        print("Please try again")
        quit()
    
    # Check if a valid movie was selected before proceeding
    if index is None:
        print("No valid movie selected. Please restart the program.")
        quit()
    
    print("-"*60)
    while True:
        print("Thank you for your input, just to confirm here is your title ", titles)
        tm.sleep(1)
        yesOrNo = input("y/n: ")
        if yesOrNo == "y":
            print("-"*60)
            print("...Searching for movies in the database...")
            print("may take some time...please wait...")
            tm.sleep(1)
            print("-"*60)
            vectoriser(index)
            break
            
            
        elif yesOrNo == "n":
            print("Sorry Must Restart the program")
            print("-"*60)
            print("Restarting....")
            tm.sleep(2)
            quit()
        else:
            print("Invalid input")
            continue

                              # // pipeline vectorize text features  , then add all the numerical features ,   scale all features  , then put into matrix _--> feed into model // train // predict user inputs
def vectoriser(index):
    # creates extra column  for text features combined in the dataFrame
    filterDF["textFeaturesCombined"] = (filterDF["overview"].fillna("") + " " + filterDF["tagline"].fillna("") + " " + filterDF["keywords"].fillna("")*5 + " " + (filterDF["genres"].fillna("") + " ") * 5 # 5 times for more weight
    )
    vectoriser = TfidfVectorizer(max_features=7000, stop_words="english")  # max_features is the number of different features / words to compare  so 3000 of love , fight , action .... so on 
    textMatrix = vectoriser.fit_transform(filterDF["textFeaturesCombined"]) # vocab for ALL MOVIES of textFeaturesCombined
    indexForUserTitle = index
    movieSpecifictext = filterDF.loc[indexForUserTitle, "textFeaturesCombined"] # find the column with the text features combines for USER SELECTEDmovie
    movieMatrix = vectoriser.transform([movieSpecifictext])  # not fit only transform to use the vocab OF ALL movies
    numberFeatureScaler(movieMatrix, indexForUserTitle,vectoriser,textMatrix)  # transport 
    
def numberFeatureScaler(movieMatrix, indexForUserTitle,vectoriser,textMatrix):
    filterDF["adult"] = filterDF["adult"].fillna(False).astype(int)  # fills NaN with false and converts current to corresponding nmbr 0 or 1
    numberFeatures = ["vote_average", "vote_count", "revenue","popularity","budget","runtime","adult","ROI"]
    filterDF[numberFeatures] = filterDF[numberFeatures].fillna(0)
    allNumberFeatures = filterDF[numberFeatures].values
    scaler = MinMaxScaler()
    allNumberFeaturesScaled = scaler.fit_transform(allNumberFeatures)  # fit and transform all the numeric values of all movies in df
    
   
    movieSpecificNmbrFeatures = filterDF.loc[indexForUserTitle, numberFeatures].values.reshape(1, -1) #numpy into 2D array  scaler.transform NEEDS a 2d array 
    movieSpecificNmbrFScaled = scaler.transform(movieSpecificNmbrFeatures) # transform the USERS numeric movies values
    
    
    userMovieMatrix = hstack([movieMatrix, csr_matrix(movieSpecificNmbrFScaled)]) # Users movie Matrix NEEDED as INPUT and PREDICT # csr reduces memory usage
    fullMatrix = hstack([textMatrix, csr_matrix(allNumberFeaturesScaled)])    # NEEDED for model to TRAIN ON 
    recommender(userMovieMatrix, indexForUserTitle, fullMatrix)


def recommender(userMovieMatrix, indexForUserTitle, fullMatrix):
    kNNmodel = NearestNeighbors(n_neighbors=8, metric="cosine")  # cosine similarty 
    kNNmodel.fit(fullMatrix) # train with all inputs 
    distances, indices = kNNmodel.kneighbors(userMovieMatrix)  # distance how far each neighbor is ( the less distance the closer it is to the users movie)      index is just the indices the distance is linked to
    for i, dist in zip(indices[0][1:8], distances[0][1:8]):  
        similarity = (1 - dist)*100  # convert distance to similarity
        print(filterDF.iloc[i]["title"], "  Similarity %:", str(similarity) + "%")

    while True:
      retry = input("Retry press r or q to quit")
      if retry == "r":
          main()
          break
      elif retry == "q":
          quit()
      else:
        print("Invalid input")
        continue

main()


    