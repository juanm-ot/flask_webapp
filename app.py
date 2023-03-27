from flask import Flask, render_template,request
import os
import make_plot
import matplotlib

matplotlib.use('Agg')
app = Flask(__name__, static_url_path='/static')


app.config["SECRET_KEY"] = "you-will-never-guess"


# @app.route("/", methods=["GET", "POST"])
# @app.route("/index", methods=("GET", "POST"))



@app.route("/", methods=["GET", "POST"])
def index():
    members = ['Gabriel Jaime Orrego Restrepo // CC 1036963928 - gorrego@unal.edu.co', 'Santiago Mendoza Mejia // CC 1020473888 - smendozam@unal.edu.co', 'Diego Alexander Ortiz Rua // CC 1152456783 - diaortizru@unal.edu.co', 'Carolina Herrera Arredondo // CC 1053872140 - caherreraar@unal.edu.co','Juan Manuel Ramírez Otálvaro // CC 1040046257 - jumramirezot@unal.edu.co']
    make_plot.prediction_plot()
    return render_template("index.html",members=members)

@app.route('/pagina2', methods=['GET', 'POST'])
def pagina2():
    if request.method == 'POST':
        opcion = request.form['opcion']
        secciones = {
            'Modelar para clasificar': 'Modelar para clasificar',
            'Regresión Logística': 'Regresión Logística',
            'Proceso de modelado': 'Proceso de modelado',
            'Predicción - Clasificación': 'Predicción - Clasificación'
        }
        seccion = secciones.get(opcion, None)
        print(seccion)
        return render_template('pagina2.html', seccion=seccion,secciones=secciones)
    return render_template('pagina2.html', seccion=seccion,secciones=secciones)



if __name__ == "__main__":
    app.run(debug=True, port=5006,host="0.0.0.0")



