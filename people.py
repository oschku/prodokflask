from datetime import datetime

def get_timestamp():
    return datetime.now().strftime(("%Y-%m-%d %H:%m:%S"))

#Data to serve with our API
PEOPLE = {
    "Moe": {
        "fname": "Momodou",
        "lname": "Krubally",
        "timestamp": get_timestamp()
    },
    "Oskchu": {
        "fname": "Oskari",
        "lname": "Honkasalo",
        "timestamp": get_timestamp()
    },
    "Aleks": {
        "fname": "Aleksi",
        "lname": "Ojala",
        "timestamp": get_timestamp()
    },
    "Walt": {
        "fname": "Waltteri",
        "lname": "Naapuri",
        "timestamp": get_timestamp()
    }
}
#create read handler
def read():
    return [PEOPLE[key] for key in sorted(PEOPLE.keys())]