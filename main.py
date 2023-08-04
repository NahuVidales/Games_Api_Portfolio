from fastapi import FastAPI
import pandas as pd
import uvicorn
import ast


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
@app.get('/genero/{Año}')
def genero(Año: int):
    gamesYear = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == Año:
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
@app.get('/juegos/{Año}')
def juegos( Año: int ):
    games_in_year = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == Año:
            games_in_year.append(steam.iloc[i]['title'])
    return games_in_year


#def specs( Año: str ): Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente.
@app.get('/specs/{Año}')
def specs(Año: int):
    specsYear = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == Año:
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
def earlyacces(Año: int):
    earlyYear = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == Año:
            earlyYear.append(steam.iloc[i]['early_access'])
    return print('In the', Año, 'were released', earlyYear.count(True), 'games with early access')




#def sentiment( Año: str ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.
@app.get('/sentiment/{Año}')
def sentiment(Año: int):
    sentimentYear = []
    for i in range(len(steam['year'])):
        if steam['year'][i] == Año:
            sentimentYear.append(steam.iloc[i]['sentiment'])
    
    #evaluaremos cada genero y lo guardaremos en un diccionario
    sentiment = {}
    for i in range(len(sentimentYear)):
        if sentimentYear[i] in sentiment:
            sentiment[sentimentYear[i]] += 1
        else:
            sentiment[sentimentYear[i]] = 1
    return sentiment


#def metascore( Año: str ): Top 5 juegos según año con mayor metascore.
@app.get('/metascore/{Año}')
def metascore(Año: int):
    # Filter the DataFrame to get rows with the specified year
    steamYear = steam[steam['year'] == Año]
    
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


