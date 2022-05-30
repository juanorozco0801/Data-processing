# %%
import numpy as np
import skfuzzy as fuzz # liberia para el diseño del controlador difuso -> pip install sickit-fuzzy
from skfuzzy import control as ctrl
#import jupyter
import matplotlib.pyplot as plt
# %%
# Se definen las condiciones de las entradas y salidas del sistema

"""

"""
#entradas del sistema donde defino velocidad del vehículo y distancia al siguiente obstaculo 

Speed = ctrl.Antecedent(np.arange(0, 300, 1), 'speed')
Distance = ctrl.Antecedent(np.arange(0, 100, 1), 'distance')
#Acceleration = ctrl.Antecedent(np.arange(0, 500, 1), 'acceleration') 

#salidas del sistema se define fuerza de frenado

BrakingForce = ctrl.Consequent(np.arange(0, 100, 1), 'brakingForce')

# Generación automática de las funciones de pertenencia de la población
Speed.automf(5)
Distance.automf(5)
#Acceleration.automf(5)

# Construcción de las funciones de pertenencia  de manera personalizada
BrakingForce['bajo'] = fuzz.trimf(BrakingForce.universe, [0, 0, 30])
BrakingForce['medioBajo'] = fuzz.gaussmf(BrakingForce.universe, mean=40, sigma=8)
BrakingForce['medioAlto'] = fuzz.gaussmf(BrakingForce.universe, mean=60, sigma=8)
BrakingForce['alto'] = fuzz.trimf(BrakingForce.universe, [70, 100, 100])


# Para conocer más funciones de pertenencia visitar: http://pythonhosted.org/scikit-fuzzy/api/skfuzzy.membership.html
# Se grafican las funciones de transferencia
# %%
Speed.view()
plt.show()
# %
Distance.view()
plt.show()
# %%
#Acceleration.view()
#plt.show()
# %%
BrakingForce.view()
plt.show()
# %%
# Se crean las reglas de inferencia para el problema

rule1 = ctrl.Rule((Speed['good'] & Distance['mediocre']) | Distance['poor'], BrakingForce['alto'])

rule2 = ctrl.Rule((Speed['decent'] & Distance['mediocre']) | (Speed['average'] & Distance['mediocre']) , BrakingForce['medioAlto'])

rule3 = ctrl.Rule(Speed['mediocre'] & Distance['decent'] | (Speed['average'] & Distance['decent']), BrakingForce['medioBajo'])

rule4 = ctrl.Rule(Speed['poor'] & Distance['good'] , BrakingForce['bajo'])


rule2.view()
plt.show()

# %%
# Se crea el controlador del sistema

x=0 

while x == 0:
    
    braking_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    # Se realiza una simulación dle controlador para una situación en específico
    braking = ctrl.ControlSystemSimulation(braking_ctrl)
    
    # Se dan valores a las entradas del sistema

    braking.input['speed'] = int(input('Ingrese la velocidad del vehiculo (km/h): '))
    braking.input['distance'] = int(input('Ingrese distancia al vehiculo proximo (m): '))
    #braking.input['acceleration'] = int(input('Ingrese aceleracion del vehiculo (m/s^2): '))


    # Se procesan los datos y se obtiene el resultado
    braking.compute()
    print(braking.output['brakingForce'])
    BrakingForce.view(sim=braking)
    plt.show()

    
    x = int(input('Desea continuar? SI (presione 0) o NO (presione 2): '))


