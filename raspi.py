from flask import Flask
from gpiozero import Servo
from time import sleep

servo = Servo(12)

# Guardem la posició actual (comencem a 90° = centre)
posicio_actual = 90

def moure_relatiu(delta):
    global posicio_actual

    posicio_actual += delta  # suma o resta graus

    # limitar entre 0 i 180
    if posicio_actual < 0:
        posicio_actual = 0
    elif posicio_actual > 180:
        posicio_actual = 180

    # convertir a valor (-1 a 1)
    valor = (posicio_actual / 90) - 1
    servo.value = valor

app = Flask(__name__)

@app.route("/")
def home():
    return "Servidor funcionant"

@app.route("/dreta")
def dreta():
    moure_relatiu(2)
    print("Moviment cap a la dreta")
    return "OK"

@app.route("/esquerra")
def esquerra():
    moure_relatiu(-2)
    print("Moviment cap a l'esquerra")
    return "OK"

if __name__ == "__main__":
    print("Arrencant servidor...")
    app.run(host="0.0.0.0", port=5001)
