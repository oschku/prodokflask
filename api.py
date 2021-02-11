from flask import Flask, render_template
import connexion
from flask_swagger_ui import get_swaggerui_blueprint
from project import create_app

#create the application instance
#app = Flask(__name__, template_folder="templates")

#let connexion handle the initiation of app
# app = connexion.App(__name__, specification_dir='./')


# #force app to read swagger.yml file to configure the endpoints
# app.add_api('specfile.yml')

# SWAGGER_URL = '/api/ui'

#url home route
app = create_app()



if __name__ == "__main__":
    app.run(host = '0.0.0.0', port= 5000,  debug=True)