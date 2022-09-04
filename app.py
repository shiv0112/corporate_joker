from flask import Flask,request,render_template
from joke.exception import JokeException
from joke.logger import logging
import pyjokes

app = Flask(__name__)

@app.route('/')
def index():
    try:
        logging.info("Home page opened")
        return render_template('index.html',methods=['GET','POST'],embed="")
        
    except Exception as e:
        raise JokeException(e,sys) from e

@app.route('/random_joke')
def result():
    try:
        return render_template('result1.html',methods=['GET','POST'],embed=pyjokes.get_joke(language='en', category='all'))
    except Exception as e:
        raise JokeException(e,sys) from e

@app.route('/joke_generated')
def result():
    try:
        return render_template('result2.html',methods=['GET','POST'],embed=pyjokes.get_joke(language='en', category='all'))
    except Exception as e:
        raise JokeException(e,sys) from e

@app.route('/team')
def team():
    try:
        return render_template('team.html',methods=['GET','POST'])
    except Exception as e:
        raise JokeException(e,sys) from e

@app.route('/about')
def about():
    try:
        return render_template('about.html',methods=['GET','POST'])
    except Exception as e:
        raise JokeException(e,sys) from e

if __name__=="__main__":
    app.run(debug=True)
