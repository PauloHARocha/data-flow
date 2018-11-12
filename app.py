import json
from flask import Flask, send_file, request
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

@app.route("/experiment/<exp>/algorithm/<alg>/<k>/clusters")
def PlotClusters(exp, alg, k):
    plot = Plot(experiment=str(exp))
    res = plot.plot_clusters(alg, k=int(k))
    return json.dumps(res)


metrics = ['inter-cluster', 'cluster-separation', 'abgss',
           'edge-index', 'cluster-connectedness', 'intra-cluster',
           'ball-hall', 'intracluster-entropy', 'ch-index', 'hartigan',
           'xu-index', 'wb-index', 'dunn-index', 'davies-bouldin', 'cs-measure',
           'silhouette', 'min-max-cut']


@app.route("/experiment/<exp>/metrics", methods=['GET', 'POST'])
def PlotMetrics(exp):
    algorithms = request.get_json()
    kmin= min(algorithms[0]['k'])
    kmax= max(algorithms[0]['k'])
    plot = Plot(experiment=str(exp), 
                algorithms=[alg['value'] for alg in algorithms], 
                metrics=metrics, k_min=kmin, k_max=kmax)

    res = plot.plot_k_range()
    return json.dumps(res)


@app.route("/experiment/<exp>/metrics/corr", methods=['GET', 'POST'])
def PlotCorrMatrix(exp):
    algorithms = request.get_json()
    kmin= min(algorithms[0]['k'])
    kmax= max(algorithms[0]['k'])
    plot = Plot(experiment=str(exp), 
                algorithms=[alg['value'] for alg in algorithms],
                metrics=metrics, k_min=kmin, k_max=kmax)
    filename = plot.gen_corr_df()
    return send_file(filename, mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)
