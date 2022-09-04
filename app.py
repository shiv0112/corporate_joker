from flask import Flask,request,render_template

app = Flask(__name__)

@app.route('/')
def index():
    try:
        return render_template('index.html',methods=['GET','POST'])
    except Exception as e:
        return str(e)

@app.route('/team')
def team():
    try:
        return render_template('team.html',methods=['GET','POST'])
    except Exception as e:
        return str(e)



if __name__=="__main__":
    app.run(debug=True)
