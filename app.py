from flask import Flask, request, jsonify
import logging
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

logging.basicConfig(filename='demo.log', level=logging.INFO)



def scale(payload):
    """Scales Payload"""
    
    scaler = StandardScaler().fit(payload.astype(float))
    scaled_adhoc_predict = scaler.transform(payload.astype(float))
    return scaled_adhoc_predict

@app.route("/")
def home():
    html = f"<h3>Sklearn Prediction Home</h3>"
    return html.format(format)

@app.route("/predict", methods=['POST'])
def predict():
    """Performs an sklearn prediction
        
        input looks like:
        {
        "CHAS":{
        "0":0
        },
        "RM":{
        "0":6.575
        },
        "TAX":{
        "0":296.0
        },
        "PTRATIO":{
        "0":15.3
        },
        "B":{
        "0":396.9
        },
        "LSTAT":{
        "0":4.98
        }
        
        result looks like:
        { "prediction": [ <val> ] }
        
        """
    #Logging status
    
    # Logging the input payload
    json_payload = request.json
    app.logger.info(f"JSON payload: \n{json_payload}")
    inference_payload = pd.DataFrame(json_payload)
    app.logger.info(f"Inference payload DataFrame: \n{inference_payload}")
    # scale the input
    scaled_payload = scale(inference_payload)
    # get an output prediction from the pretrained model, clf
    prediction = list(clf.predict(scaled_payload))
    # TO DO:  Log the output prediction value
    app.logger.info(f"Prediction: \n{prediction}")
    return jsonify({'prediction': prediction,})

if __name__ == "__main__":
    # load pretrained model as clf
    clf = joblib.load("./model_data/boston_housing_prediction.joblib")
    app.run(host='0.0.0.0', port=80, debug=True) # specify port=80
