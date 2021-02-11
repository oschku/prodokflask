from datetime import datetime
from project import valuation
import json
from project.models import UserInput
from project.models import db


def get_timestamp():
    return datetime.now().strftime(("%Y-%m-%d %H:%m:%S"))


def run(
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

    luokat = valuation.kuntonumerot
    data.update({'kunto' : luokat.get(data('kunto'))})


    ui_result = UserInput(**data)
    db.session.add(ui_result)
    db.session.commit()


    return data




def input(
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
    id,
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
    'id': id, 
    'sauna': sauna
    }
    

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