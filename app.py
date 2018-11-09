import json
from flask import Flask
from flask_cors import CORS
from API.plotJson import Plot

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '<h1>Data Flow api</h1>'


@app.route("/experiments")
def Experiments():
    with open('API/experiments.json') as f:
        data = json.load(f)
        f.close()
    return json.dumps(data)


@app.route("/experiment/<exp>/plot")
def PlotDistribution(exp):
    plot = Plot(experiment=str(exp))
    res = plot.plot_data_distribution()
    return json.dumps(res)

if __name__ == '__main__':
    app.run(debug=True)
