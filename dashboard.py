
import geopandas
import pandas as pd
import streamlit as st
import numpy as np
import folium
import plotly
import plotly.express as px

from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster

from datetime import datetime

server = dashboard.server
st.set_page_config(layout='wide')


@st.cache(allow_output_mutation=True)
def get_data(path):
    df = pd.read_csv(path)
    return df

@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    geofile = geopandas.read_file( url )
    return geofile

def set_feature(data):
    #add new features
    df['price_m2'] = df['price'] / df['sqft_lot']

    return df

def overview_data(df):
    f_attributes = st.sidebar.multiselect('Enter columns', df.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', df['zipcode'].unique())

    st.title('Data Overview')

    if (f_zipcode != []) & (f_attributes != []):
        df = df.loc[df['zipcode'].isin(f_zipcode), f_attributes]

    elif (f_zipcode != []) & (f_attributes == []):
        df = df.loc[df['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
            df = df.loc[:, f_attributes]

    else:
        df = df.copy()


    st.dataframe(df)
    #Definindo a quantidade de colunas no streamlit

    c1, c2 = st.columns((1, 1)) # Define o tamanho da coluna

    #attributes + zipcode = Selecionar Colunas e Linhas
    #attributes = Selecionar colunas
    #zipcode = Selecionar Linhas
    #Average metrics

    df1 = df[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = df[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = df[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()



    #merge

    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df_copy1 = pd.merge(m2, df4, on='zipcode', how='inner')

    df_copy1.columns = ['ZIPCODE', 'TOTAL HOUSE', 'PRICE', 'SQRT LIVING', 'PRICE/m2']

    c1.header('Average Values')
    c1.dataframe(df_copy1, height=600)
    #st.dataframe(df.head())

    #Statistic Descriptive

    num_attributes = df.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df_copy2 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()

    df_copy2.columns = ['Attributes', 'Max', 'Min', 'Mean', 'Median', 'Std']

    c2.header( 'Descriptive Analysis' )
    c2.dataframe(df_copy2, height=600)

    return None

def portifolio_density(df, geofile):
    st.title('Region Overview')

    c1, c2 = st.columns((1, 1))
    c1.header('Portifolio Density')

    df1 = df.copy()
    df_test = df1.sample(10)


    #Base Map - Folium
    density_map = folium.Map(location=[df_test['lat'].mean(),
                df_test['long'].mean()], 
                default_zoom_starts=15)

    marker_cluster = MarkerCluster().add_to( density_map )
    for name, row in df_test.iterrows():
        folium.Marker( [row['lat'], row['long'] ], 
            popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format( row['price'],
                                        row['date'],
                                        row['sqft_living'],
                                        row['bedrooms'],
                                        row['bathrooms'],
                                        row['yr_built'] ) ).add_to( marker_cluster )

    with c1:
        folium_static(density_map)


    #Region Price Map
    c2.header('Price Density')

    df_copy3 = df_test[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df_copy3.columns = ['ZIP', 'PRICE']


    geofile = geofile[geofile['ZIP'].isin(df_copy3['ZIP'].tolist())]

    region_price_map = folium.Map( location=[df_test['lat'].mean(), 
                                df_test['long'].mean() ],
                                default_zoom_start=15 ) 


    region_price_map.choropleth( data = df_copy3,
                                geo_data = geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity = 0.7,
                                line_opacity = 0.2,
                                legend_name='AVG PRICE' )

    with c2:
        folium_static( region_price_map )

    return None


def commercial(df):
    st.sidebar.title('Comercial Options')
    st.title('Commercial Atrributes')

    #-------Filtros

    min_year_built = int(df['yr_built'].min())
    max_year_built = int(df['yr_built'].max())
    mean_year_built = int(df['yr_built'].mean())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built',min_year_built,
                                                max_year_built,
                                                mean_year_built)



    #-------Avarege Price per Year Built


    df_copy4 = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
    df_copy4 = df_copy4.loc[df_copy4['yr_built'].astype(int)<f_year_built]

    fig = px.line(df_copy4, x='yr_built', y='price')

    st.plotly_chart(fig, use_container_width=True)

    #-------Avarege Price per Day

    st.header('Avarege Price per Day')
    st.sidebar.subheader('Select Max Date')

    #filters
    df['date'] = pd.to_datetime( df['date'] ).dt.strftime( '%Y-%m-%d' )

    min_date = datetime.strptime(df['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(df['date'].max(), '%Y-%m-%d')



    f_date = st.sidebar.slider('Date', min_date, max_date, max_date)


    #data filtering
    df['date'] = pd.to_datetime(df['date'])

    df_copy5 = df[['date', 'price']].groupby('date').mean().reset_index()
    df_copy5 = df_copy5.loc[df_copy5['date'] < f_date]

    fig = px.line(df_copy5, x='date', y='price')

    st.plotly_chart(fig, use_container_width=True)

    #-------------Histograma

    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    #filter
    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    avg_price = int(df['price'].mean())

    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df_copy5 = df[df['price'] < f_price]
    
    st.write(f'Foram encontrados {df_copy5.shape[0]} imóveis')

    # data plot
    fig = px.histogram (df_copy5, x= 'price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution(df):

    #=====================================================
    #Distribução dos imoveis por categorias fisicas
    #=====================================================

    st.sidebar.title('Atributes Options')
    st.title('House Attributes')

    #filters
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms',
                                    sorted(set(df['bedrooms'].unique())))

    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms',
                                    sorted(set(df['bathrooms'].unique())))

    f_floors =st.sidebar.selectbox('Max number of floors',
                                    sorted(set(df['floors'].unique())))


    c1, c2 = st.columns(2)  

    # House per bedrooms
    c1.header('Houses per bedrooms')
    df_bed = df[df['bedrooms']<= f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)


    # House per bathrooms
    c2.header('Houses per bathrooms')
    df_bath = df[df['bathrooms']<= f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # House per flors
    st.header('Houses per floors')
    df_flors = df[df['floors'] <= f_floors]
    fig = px.histogram(df, x='floors', nbins=10)
    st.plotly_chart(fig, use_container_width=True)

    return None


if __name__ == '__main__':

#ETL
#data extration
path = 'kc_house_data/data.csv'
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'


df = get_data(path)
geofile = get_geofile( url )


#transformation
df = set_feature(df)

overview_data(df)

portifolio_density(df, geofile)

commercial(df)

attributes_distribution(df)
    
    #if __name__ == "__main__":
    #dashboard.run_server(debug=False)
