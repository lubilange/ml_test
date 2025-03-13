import pickle
import pandas as pd   
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
db=pd.read_csv('zara.csv', sep=';') 
db.columns=db.columns.str.replace(' ','')
 
with open('zara.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as s:
    scaler = pickle.load(s)

st.title("ML POUR LA RENTABILITE D'UN PRODUIT VENDU EN MAGASIN")
ProductPosition = st.selectbox("Position du produit en magasin", db['ProductPosition'].unique())
Seasonal = st.selectbox("Produit saisonnier", db['Seasonal'].unique())
SalesVolume = st.number_input("Volume de vente", min_value=1, value=2)
price = st.number_input("Prix", min_value=1.0, value=1.00)
term = st.text_input("Vetement", "")
section = st.selectbox("Section (MAN/WOMAN)", db['section'].unique())

if not term or ProductPosition is None or Seasonal is None or SalesVolume is None or price is None or section is None:
    st.error("Tous les champs doivent être remplis !")
else:
    # Affichage des valeurs saisies
    st.write(f"Position du produit : {ProductPosition}")
    st.write(f"Produit saisonnier : {Seasonal}")
    st.write(f"Volume de vente : {SalesVolume}")
    st.write(f"Prix : {price}")
    st.write(f"Vêtement: {term}")
    st.write(f"Section : {section}")
db['revenu'] = db['price'] * db['SalesVolume']
revenu = price * SalesVolume
def seuil_rentabilite(x):
    seuil = 100000  
    
    if x['Seasonal'] == 1:
        seuil *= 0.9  # Réduire le seuil pour les produits saisonniers
    
    if x['ProductPosition'] == 1:
        seuil *= 1.2  # Augmenter le seuil pour les produits en fin de rayon

    if x['section'] == 1:  # Si c'est un produit féminin (exemple)
        seuil *= 1.1  # Augmenter encore pour les produits dans la section "femme"
    
    return seuil

db['rentable'] = db.apply(lambda x: 1 if x['revenu'] > seuil_rentabilite(x) else 0, axis=1)

encoders = {}
for col in ['section', 'Seasonal', 'ProductPosition']:
    encoders[col] = LabelEncoder()
    db[col] = encoders[col].fit_transform(db[col])


input_data = pd.DataFrame({
    'ProductPosition': [ProductPosition],
    'Seasonal': [Seasonal],
    'SalesVolume': [SalesVolume],
    'price': [price],  
    'section': [section],    
    'revenu': [revenu]
})

input_data['section'] = encoders['section'].transform(input_data['section'])
input_data['Seasonal'] = encoders['Seasonal'].transform(input_data['Seasonal'])
input_data['ProductPosition'] = encoders['ProductPosition'].transform(input_data['ProductPosition'])

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
probabilities = model.predict_proba(input_data_scaled)


if prediction[0] == 1:
    if not term or ProductPosition is None or Seasonal is None or SalesVolume is None or price is None or section is None:
        st.error("pas de prédiction sans données !")
    else:
        st.write("Le produit sera **non rentable**.")
else:
    if not term or ProductPosition is None or Seasonal is None or SalesVolume is None or price is None or section is None:
        st.error("pas de prédiction sans données")
    else:    
     st.write("Le produit sera **non rentable**.")

     if prediction[0] == 0:
            st.write("Suggestions pour rendre le produit plus rentable :")
            
            if price < 150:
                st.write("Augmenter le prix du produit pourrait améliorer la rentabilité. Essayer un prix supérieur à 150.")
            
            if SalesVolume < 200:
                st.write("Augmenter le volume de vente en mettant en place des promotions ou des campagnes marketing.")
            
            if Seasonal == 0:  
                st.write("Envisager de rendre ce produit saisonnier ou de le promouvoir davantage pendant les saisons clés.")
            
            if ProductPosition == 0:  
                st.write("Améliorer la visibilité du produit en le plaçant dans une position plus stratégique dans le magasin.")
            
            if probabilities[0][1] < 0.5:
                st.write("La probabilité de rentabilité est faible. Considérer une révision complète de la stratégie marketing et de vente.")
    