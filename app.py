###Run by ANIRUDH CHANDNANI


from flask import Flask,render_template,request 
import pickle
from tensorflow.keras.models import load_model

###Loading model and cv
cv = pickle.load(open('cv.pkl','rb')) ##loading cv
model = load_model('bilstm.h5') ##loading model

app = Flask(__name__) ## defining flask name

@app.route('/') ## home route
def home():
    return render_template('index.html') ##at home route returning index.html to show

def func(x):
    if x>0.5:
        return 1
    else:
        return 0
    
def pred_fn(x):
    for i in range (len(x)):
        x[i] = func(x[i])
    return x[0]
    
@app.route('/predict',methods=['POST']) ## on post request /predict 
def predict():
    if request.method=='POST':     
        mail = request.form['email']  ## requesting the content of the text field
        data = [mail] ## converting text into a list
        vect = cv.transform(data).toarray() ## transforming the list of sentence into vecotor form
        y_pred = model.predict(vect) ## predicting the class(1=spam,0=ham)
        #return render_template('result.html',prediction=pred) ## returning result.html with prediction var value as class value(0,1)
        pred = pred_fn(y_pred)       
        
        if pred==1:
            return render_template('index.html',prediction_text="SPAM")
        else:
            return render_template('index.html',prediction_text="NOT SPAM")
if __name__ == "__main__":
    app.run(debug=True)     ## running the flask app as debug==True
