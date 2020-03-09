from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/')
def hello():
    return 'hello world'


@app.route('/predict', methods=['GET'])
def hello_world():
    from sklearn.externals import joblib
    model = joblib.load('iris.ml')
    iris_predict = model.predict([[5.9,3.0,5.1,1.8]])
    return str(iris_predict[0])


if __name__ == '__main__':
    app.run(debug=True)
