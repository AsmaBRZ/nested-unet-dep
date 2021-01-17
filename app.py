from flask import Flask,request, render_template
from model.prediction import *

app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
   return render_template('index.html')


@app.route('/photoRecognize', methods=['POST'])
def photoRecognize():
    if request.method == 'POST': 
        data = request.files['image_data']
        if data == None:
            return 'no image received'
        else:
            # model.predict.predict returns a dictionary
            prediction = predict(data)
    else:
      return render_template('index.html')

    return jsonify(results=prediction)


if __name__ == '__main__':
   app.run(debug = True)