from datetime import datetime as dtm
from project import valuation
import json
from project.models import db, UserInput, ApiKey
from sqlalchemy import text
import pandas as pd
from dateutil import tz 
import pytz
import datetime

def convert_datetime_timezone(dt, tz1, tz2):
    tz1 = pytz.timezone(tz1)
    tz2 = pytz.timezone(tz2)

    dt = datetime.datetime.strptime(dt,"%d.%m.%Y %H:%M:%S")
    dt = tz1.localize(dt)
    print(dt)
    dt = dt.astimezone(tz2)
    dt = dt.strftime("%d.%m.%Y %H:%M:%S")

    return dt



def get_timestamp():
    return dtm.now().strftime(("%Y-%m-%d %H:%m:%S"))


def run(
    apiKey,
    osoite,
    kerros,
    parveke,
    kunta,
    kerros_yht,
    tonttiala,
    postinumero,
    kunto,
    muu_kerrosala,
    asuntotyyppi,
    tontti,
    created_on,
    asuinala,
    vastike,
    user,
    rakennusvuosi,
    vuokrattu,
    query_id,
    huone_lkm,
    hissi,
    sauna
):
    data = {
    'osoite': osoite, 
    'kerros': kerros, 
    'parveke': parveke, 
    'lat': None, 
    'kunta': kunta, 
    'kerros_yht': kerros_yht, 
    'tonttiala': tonttiala, 
    'lng': None, 
    'postinumero': postinumero, 
    'kunto': kunto, 
    'muu_kerrosala': muu_kerrosala, 
    'asuntotyyppi': asuntotyyppi, 
    'tontti': tontti, 
    'created_on': created_on, 
    'asuinala': asuinala, 
    'vastike': vastike, 
    'user': user, 
    'rakennusvuosi': rakennusvuosi, 
    'vuokrattu': vuokrattu, 
    'query_id': query_id, 
    'huone_lkm': huone_lkm, 
    'hissi': hissi, 
    'hinta': None, 
    'sauna': sauna
    }


    
    current_user = int(data.get('user'))
    created_on = convert_datetime_timezone(data.get('created_on'), "Europe/Stockholm", "UTC")    
    print(created_on)
    created_on = datetime.datetime.strptime(created_on, '%d.%m.%Y %H:%M:%S')
    data.update({'created_on':created_on})

    sql = (f'SELECT apikey FROM public.api_keys \
        WHERE api_keys.user_id = {current_user} \
                ORDER BY user_id ASC   ')

    with db.engine.connect() as connection:
        results = pd.read_sql(sql = sql, con = connection)
        key = results.iloc[0,0]
    
    
    if apiKey != key:
        response = {'Auth error': 'API Key not valid'}, 400
        return response
    else:

        try:
            hinta, dataset = valuation.calculate(data, query_id)
            hinta = float(hinta)
            hinta = round(hinta, -3)
            lat,lng = valuation.geodata.geocode(osoite, kunta)

            data.update({'hinta': hinta})
            data.update({'lat':lat})
            data.update({'lng':lng})
            
            data.pop('rak_ika')
            data.pop('uudiskohde')
            data.pop('hoitovastike_per_nelio')

            luokat = valuation.ui_input.kuntonumerot
            data.update({'kunto' : luokat.get(data.get('kunto'))})

            ui_result = UserInput(**data)
            db.session.add(ui_result)
            db.session.commit()


            return data
            
        
        except ValueError as V:
            
            response = {}
            data.pop('rak_ika')
            data.pop('uudiskohde')
            data.pop('hoitovastike_per_nelio')

            ui_result = UserInput(**data)
            db.session.add(ui_result)
            db.session.commit()
    

            if V.args[1] == 'street':
                print('Osoite on virheellinen','street_err')
                print(V.args[1])
                response = {'Error': 'street_error', 'message':'Osoite on virheellinen'}
            elif V.args[1] == 'country':
                print('Osoite on virheellinen','input_err')
                print(V.args[1])
                response = {'Error': 'country_error', 'message':'Osoite on virheellinen'}
            elif V.args[1] == 'city':
                print( f'Kunnasta {data.get("kunta")} ei löytynyt osoitetta {data.get("osoite")}. Tarkista kunta','city_err')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'city_error', 'message':f'Kunnasta {data.get("kunta")} ei löytynyt osoitetta {data.get("osoite")}. Tarkista kunta'}
            elif V.args[1] == 'bad_score':
                print( f'Osoitteella {data.get("osoite")} epäselvä osumatulos. Kokeile toista osoitetta','street_err')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'bad_score', 'message':f'Osoitteella {data.get("osoite")} epäselvä osumatulos. Kokeile toista osoitetta'}
            elif V.args[1] == 'multiple_streets':
                print( f'Osoite {data.get("osoite")} tuotti virheellisen tuloksen. Tarkenna osoitteen numeroa haussa','street_err')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'number_error', 'message': f'Osoite {data.get("osoite")} tuotti virheellisen tuloksen. Tarkenna osoitteen numeroa haussa'}
            elif V.args[1] == 'no_streets':
                print( f'Osoite on virheellinen','street_number')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'street_not_found', 'message':'Kyseistä osoitetta ei löydy'}
            elif V.args[1] == 'street_number':
                print( f'Osoite {data.get("osoite")} tuotti virheellisen tuloksen. Tarkenna osoitteen numeroa haussa','street_err')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'number_error', 'message': f'Osoite {data.get("osoite")} tuotti virheellisen tuloksen. Tarkenna osoitteen numeroa haussa'}
            
            


            return response, 406







def input(
    apiKey,
    osoite,
    kerros,
    parveke,
    kunta,
    kerros_yht,
    tonttiala,
    postinumero,
    kunto,
    muu_kerrosala,
    asuntotyyppi,
    tontti,
    created_on,
    asuinala,
    vastike,
    user,
    rakennusvuosi,
    vuokrattu,
    query_id,
    huone_lkm,
    hissi,
    sauna
):
    data = {
    'osoite': osoite, 
    'kerros': kerros, 
    'parveke': parveke, 
    'lat': None, 
    'kunta': kunta, 
    'kerros_yht': kerros_yht, 
    'tonttiala': tonttiala, 
    'lng': None, 
    'postinumero': postinumero, 
    'kunto': kunto, 
    'muu_kerrosala': muu_kerrosala, 
    'asuntotyyppi': asuntotyyppi, 
    'tontti': tontti, 
    'created_on': created_on, 
    'asuinala': asuinala, 
    'vastike': vastike, 
    'user': user, 
    'rakennusvuosi': rakennusvuosi, 
    'vuokrattu': vuokrattu, 
    'query_id': query_id, 
    'huone_lkm': huone_lkm, 
    'hissi': hissi, 
    'hinta': None,  
    'sauna': sauna
    }
    
    current_user = int(data.get('user'))
    sql = (f'SELECT apikey FROM public.api_keys \
        WHERE api_keys.user_id = {current_user} \
                ORDER BY user_id ASC   ')

    created_on = convert_datetime_timezone(data.get('created_on'), "Europe/Stockholm", "UTC")    
    print(created_on)
    created_on = datetime.datetime.strptime(created_on, '%d.%m.%Y %H:%M:%S')
    data.update({'created_on':created_on})

    with db.engine.connect() as connection:
        results = pd.read_sql(sql = sql, con = connection)
        key = results.iloc[0,0]
    
    
    if apiKey != key:
        response = {'Auth error': 'API Key not valid'}, 400
        return response
    else:
        try:
            hinta, dataset = valuation.calculate(data, query_id)
            hinta = float(hinta)
            hinta = round(hinta, -3)
            lat,lng = valuation.geodata.geocode(osoite, kunta)

            cols = dataset.columns
            values = dataset.values

            print(cols[0])
            print(values[0])

            df_dict = {}
            for col, val in zip(cols, values[0]):
                df_dict.update({col : val})

            df_dict.update({'hinta': hinta})
            df_dict.update({'lat':lat})
            df_dict.update({'lng':lng})

            #input_values = json.dumps(df_dict, indent = 4)    

            return df_dict
        
        except ValueError as V:
            
            response = {}
            data.pop('rak_ika')
            data.pop('uudiskohde')
            data.pop('hoitovastike_per_nelio')

            ui_result = UserInput(**data)
            db.session.add(ui_result)
            db.session.commit()
    

            if V.args[1] == 'street':
                print('Osoite on virheellinen','street_err')
                print(V.args[1])
                response = {'Error': 'street_error', 'message':'Osoite on virheellinen'}
            elif V.args[1] == 'country':
                print('Osoite on virheellinen','input_err')
                print(V.args[1])
                response = {'Error': 'country_error', 'message':'Osoite on virheellinen'}
            elif V.args[1] == 'city':
                print( f'Kunnasta {data.get("kunta")} ei löytynyt osoitetta {data.get("osoite")}. Tarkista kunta','city_err')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'city_error', 'message':f'Kunnasta {data.get("kunta")} ei löytynyt osoitetta {data.get("osoite")}. Tarkista kunta'}
            elif V.args[1] == 'bad_score':
                print( f'Osoitteella {data.get("osoite")} epäselvä osumatulos. Kokeile toista osoitetta','street_err')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'bad_score', 'message':f'Osoitteella {data.get("osoite")} epäselvä osumatulos. Kokeile toista osoitetta'}
            elif V.args[1] == 'multiple_streets':
                print( f'Osoite {data.get("osoite")} tuotti virheellisen tuloksen. Tarkenna osoitteen numeroa haussa','street_err')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'number_error', 'message': f'Osoite {data.get("osoite")} tuotti virheellisen tuloksen. Tarkenna osoitteen numeroa haussa'}
            elif V.args[1] == 'no_streets':
                print( f'Osoite on virheellinen','street_number')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'street_not_found', 'message':'Kyseistä osoitetta ei löydy'}
            elif V.args[1] == 'street_number':
                print( f'Osoite {data.get("osoite")} tuotti virheellisen tuloksen. Tarkenna osoitteen numeroa haussa','street_err')
                print(V.args[1])
                print(UserInput.osoite)
                response = {'Error': 'number_error', 'message': f'Osoite {data.get("osoite")} tuotti virheellisen tuloksen. Tarkenna osoitteen numeroa haussa'}
            
            


            return response, 406