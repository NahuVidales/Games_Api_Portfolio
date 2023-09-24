from fastapi import FastAPI
import uvicorn
import pandas as pd
import ast 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import List

app = FastAPI()


steam = pd.read_csv('dataset/steam_games.csv')

# convertimos las cadenas en valores de lista
def convert_to_list(column_str):
    try:
        return ast.literal_eval(column_str)
    except (SyntaxError, ValueError):
        # Handle the invalid data here. For example, you can return an empty list.
        return []

# Apply the conversion function to the 'genres' column
steam['genres'] = steam['genres'].apply(convert_to_list)
steam['specs'] = steam['specs'].apply(convert_to_list)
#convertimos year a tipo entero
steam['year'] = steam['year'].astype(int)

@app.get("/")
async def root():
    return {"Portfolio": "Proyect API"}



#def genero( Año: str ): Se ingresa un año y devuelve una lista con los 5 géneros más usuales en el orden correspondiente.
@app.get('/genero/{year}')
def genero(year: int):
    gamesYear = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == year:
            gamesYear.append(steam.iloc[i]['genres'])
    
    #evaluaremos cada genero y lo guardaremos en un diccionario
    generos = {}
    for i in range(len(gamesYear)):
        for j in range(len(gamesYear[i])):
            if gamesYear[i][j] in generos:
                generos[gamesYear[i][j]] += 1
            else:
                generos[gamesYear[i][j]] = 1
    
    generosOrdenados = sorted(generos.items(), key=lambda x: x[1], reverse=True)
    return generosOrdenados[:5]


#def juegos( Año: str ): Se ingresa un año y devuelve una lista con los juegos lanzados en el año.
@app.get('/juegos/{year}')
def juegos( year: int ):
    games_in_year = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == year:
            games_in_year.append(steam.iloc[i]['title'])
    return games_in_year


#def specs( Año: str ): Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente.
@app.get('/specs/{year}')
def specs(year: int):
    specsYear = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == year:
            specsYear.append(steam.iloc[i]['specs'])
    
    #evaluaremos cada genero y lo guardaremos en un diccionario
    specs = {}
    for i in range(len(specsYear)):
        for j in range(len(specsYear[i])):
            if specsYear[i][j] in specs:
                specs[specsYear[i][j]] += 1
            else:
                specs[specsYear[i][j]] = 1
    
    generosOrdenados = sorted(specs.items(), key=lambda x: x[1], reverse=True)
    return generosOrdenados[:5]


#def earlyacces( Año: str ): Cantidad de juegos lanzados en un año con early access.

@app.get('/earlyacces/')
def earlyacces(year: int):
    earlyYear = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == year:
            earlyYear.append(steam.iloc[i]['early_access'])
    return earlyYear.count(True), 'games with early access'




#def sentiment( Año: str ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.
@app.get('/sentiment/{year}')
def sentiment(year: int):
    sentimentYear = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == year:
            sentimentYear.append(steam.iloc[i]['sentiment'])
    
    #evaluaremos cada genero y lo guardaremos en un diccionario
    sentiment = {}
    for i in range(len(sentimentYear)):
        if sentimentYear[i] in sentiment:
            sentiment[sentimentYear[i]] += 1
        else:
            sentiment[sentimentYear[i]] = 1
    #eliminamos los valores nan en sentiment
    sentiment = {k: v for k, v in sentiment.items() if pd.notna(k)}
    #eliminamos todos los lugares donde contiene 'users'
    sentiment = {k: v for k, v in sentiment.items() if 'user' not in k}
    return sentiment


#def metascore( Año: str ): Top 5 juegos según año con mayor metascore.
@app.get('/metascore/{year}')
def metascore(year: int):
    # Filter the DataFrame to get rows with the specified year
    steamYear = steam[steam['year'] == year]
    
    # Check if the DataFrame is empty
    if steamYear.empty:
        return "No data available for the specified year."
    # Drop rows with missing 'metascore' values
    steamMeta = steamYear.dropna(subset=['metascore'])
    # Check if the DataFrame after dropping missing values is empty
    if steamMeta.empty:
        return "No metascore data available for the specified year."
    # Convert the 'sentiment' column to lists
    steamMeta['sentiment'] = steamMeta['sentiment'].fillna('').apply(convert_to_list)
    # Convert 'year' to integer type
    steamMeta['year'] = steamMeta['year'].astype(int)
    # Get the top 5 metascore values
    top_metascore = steamMeta.nlargest(5, 'metascore')[['title', 'metascore']]
    
    return top_metascore

# Crear una matriz TF-IDF para el texto del título de las películas
stopwords_custom = ["the", "and", "in", "of"]  # Agrega aquí tus stopwords personalizados
tfidf = TfidfVectorizer(stop_words=stopwords_custom)

# Replace NaN values with empty string
steam['title'] = steam['title'].fillna('')

# Fit the TfidfVectorizer
tfidf_matrix = tfidf.fit_transform(steam['title'])

# Calcular la similitud del coseno entre los títulos de las películas
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def recomendacion(titulo):
    # Verificar si el título está en el DataFrame
    if titulo not in steam['title'].values:
        return f"No se encontró ninguna película con el título '{titulo}'."

    # Encontrar el índice de la película con el título dado
    indices = pd.Series(steam.index, index=steam['title']).drop_duplicates()
    idx = indices[titulo]

    # Calcular las puntuaciones de similitud de todas las películas con la película dada
    sim_scores = list(enumerate(cosine_similarities[idx]))

    # Ordenar las películas por puntaje de similitud en orden descendente
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de las películas más similares (excluyendo la película dada)
    sim_scores = sim_scores[1:6]  # Obtener las 5 películas más similares
    movie_indices = [x[0] for x in sim_scores]

    # Devolver los títulos de las películas más similares
    respuesta_recomendacion = steam['title'].iloc[movie_indices].tolist()
    return {'lista recomendada': respuesta_recomendacion}

#Una vez que toda la data es consumible por la API, está lista para consumir por los departamentos de Analytics y Machine Learning,
# y nuestro EDA nos permite entender bien los datos a los que tenemos acceso, es hora de entrenar nuestro modelo de machine 
# learning para armar un modelo de predicción. El mismo deberá basarse en características como Género, Año, Metascore y/o 
# las que creas adecuadas. Tu líder pide que el modelo derive en un GET/POST en la API simil al siguiente formato:

#def predicción( genero, earlyaccess = True/False, (Variables que elijas) ): Ingresando estos parámetros, deberíamos 
# recibir el precio y RMSE