# Importar librerias 
from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd
import numpy as np

# Cargar modelo
model = joblib.load('reg_popularidad_Spotify.pkl')

#Definición API Flask:
app = Flask(__name__)
api = Api(app, version='1.0', title='Popularity Prediction API',
          description='Predice la popularidad de canciones a partir de sus características')

ns = api.namespace('predict', description='Modelo de regresión')

# ✅ Modelo de entrada para Swagger UI
input_model = api.model('InputFeatures', {
    'duration_ms': fields.Integer(required=True),
    'explicit': fields.Integer(required=True),
    'danceability': fields.Float(required=True),
    'energy': fields.Float(required=True),
    'key': fields.Integer(required=True),
    'loudness': fields.Float(required=True),
    'mode': fields.Integer(required=True),
    'speechiness': fields.Float(required=True),
    'acousticness': fields.Float(required=True),
    'instrumentalness': fields.Float(required=True),
    'liveness': fields.Float(required=True),
    'valence': fields.Float(required=True),
    'tempo': fields.Float(required=True),
    'time_signature': fields.Integer(required=True),
})

# Definir el campo de salida con nombre personalizado
resource_fields = api.model('Output', {
    'Resultado Predicción': fields.String,
})

@ns.route('/')
class PopularityApi(Resource):
    @ns.doc(params={k: 'Input feature' for k in input_model.keys()})
    @ns.marshal_with(resource_fields)
    def get(self):
        # Leer desde parámetros de URL
        data = {k: float(request.args.get(k)) for k in input_model.keys()}
        features = np.array([list(data.values())])
        prediction = model.predict(features)[0]
        return {'Resultado Predicción': f'{prediction:.2f}'}, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)