from flask import Flask, render_template, request
import numpy as np 
import pickle

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict_placement():
    Age=request.form.get('Age')
    ChestPainType_ATA=request.form.get('ChestPainType_ATA')
    ChestPainType_TA=request.form.get('ChestPainType_TA')
    ChestPainType_NAP=request.form.get('ChestPainType_NAP')
    ChestPainType_ASY=request.form.get('ChestPainType_ASY')
    RestingBP=request.form.get('RestingBP')
    Cholesterol=request.form.get('Cholesterol')
    FastingBS=request.form.get('FastingBS')
    RestingECG_Normal=request.form.get('RestingECG_Normal')
    RestingECG_ST=request.form.get('RestingECG_ST')
    RestingECG_LVH=request.form.get('RestingECG_LVH')
    ExerciseAngina_Y=request.form.get('ExerciseAngina_Y')
    Oldpeak=request.form.get('Oldpeak')
    ST_Slope_Flat=request.form.get('ST_Slope_Flat')
    ST_Slope_Down=request.form.get('ST_Slope_Down')

    #Predict
    result = model.predict(np.array([Age,ChestPainType_ATA,ChestPainType_TA,
                                     ChestPainType_NAP,ChestPainType_ASY,RestingBP
                                     ,Cholesterol,FastingBS,RestingECG_Normal,
                                     RestingECG_ST,RestingECG_LVH,ExerciseAngina_Y,
                                     Oldpeak,ST_Slope_Flat,ST_Slope_Down]).reshape(1,-1))
    
    return str(result)
if __name__ == '__main__':
    app.run(debug= True)
