from datetime import datetime
from project import valuation


def get_timestamp():
    return datetime.now().strftime(("%Y-%m-%d %H:%m:%S"))

#create read handler
# @app.route('')
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
    

    hinta = valuation.calculate(data, query_id)
    hinta = float(hinta)
    hinta = round(hinta, -3)
    lat,lng = valuation.geodata.geocode(osoite, kunta)

    data.update({'hinta': hinta})
    data.update({'lat':lat})
    data.update({'lng':lng})

    return data