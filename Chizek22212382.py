"""
Práctica 2: Sistema Cardiovascular

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Josué Chizek Espinoza
Número de control: 22212382
Correo institucional: 22212382@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Modulos (consola) y libreria para sistemas de control
#!pip install control
#!pip install slycot
import control as ctrl

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

u = np.array(pd.read_excel('signal.xlsx',header=None))
x0,t0,tend,dt,w,h = 0,0,10,1E-3,7,3.5
N = round((tend - t0)/dt) + 1
t = np.linspace (t0,tend,N)
u = np.reshape(signal.resample(u,len(t)),-1)

def cardio(Z,C,R,L):
    num=[L*R,R*Z]
    den=[C*L*R*Z,L*R+L*Z,R*Z]
    sys = ctrl.tf(num,den)
    return sys

#Funcion de transferencia: Normotenso
Z,C,R,L = 0.033,1.5,0.95,0.01
sysnormo = cardio(Z,C,R,L)  
print(f'Funcion de transferencia del normotenso: {sysnormo}')

#Funcion de transferencia: Hipotenso
Z,C,R,L = 0.02,0.25,0.6,0.005
syshipo = cardio(Z,C,R,L)  
print(f'Funcion de transferencia del hipotenso: {syshipo}')

#Funcion de transferencia: Normotenso
Z,C,R,L = 0.05,2.5,1.4,0.02
syshiper = cardio(Z,C,R,L)  
print(f'Funcion de transferencia del hipertenso: {syshiper}')

# Respuesta del sistema en lazo abierto
clr1 = np.array([230,57,70])/255
clr2 = np.array([168,218,220])/255
clr6 = np.array([145,100,60])/255

_,Pp0 = ctrl.forced_response(sysnormo,t,u,x0)
_,Pp1 = ctrl.forced_response(syshipo,t,u,x0)
_,Pp2 = ctrl.forced_response(syshiper,t,u,x0)

fg1 = plt.figure()
plt.plot(t,Pp0,'-',linewidth = 1, color = clr1,label = 'Pp(t): Normotenso')
plt.plot(t,Pp1,'-',linewidth = 1, color = clr2,label = 'Pp(t): hipomotenso')
plt.plot(t,Pp2,'-',linewidth = 1, color = clr6,label = 'Pp(t): Hipertenso')
plt.grid(False)
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4);plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t [s]', fontsize = 11)
plt.ylabel('Pp(t) [V]', fontsize = 11)
plt.legend(bbox_to_anchor = (0.5,-0.2),loc = 'center', ncol = 3,
           fontsize = 9, frameon = True)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('Sistema Cardiovascular python.png',dpi=600, bbox_inches ='tight')
fg1.savefig('Sistema Cardiovascular python.pdf')

def controlador (kP,kI,sys):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI,denPI)
    X = ctrl.series(PI, sys)
    sysPI = ctrl.feedback(X,1,sign=-1)
    return sysPI

hipoPI = controlador (1.000340631088294622,404.434384377147,syshipo)
hiperPI = controlador(10,103.569197822789,syshiper) 

_,Pp3 = ctrl.forced_response(hipoPI,t,Pp0,x0)
_,Pp4 = ctrl.forced_response(hiperPI,t,Pp0,x0)

fg2 = plt.figure()
plt.plot(t,Pp0,'-',linewidth = 1, color = clr1,label = 'Pp(t): Normotenso')
plt.plot(t,Pp3,'--',linewidth = 1, color = clr2,label = 'Pp(t): HipotensoPID')
plt.plot(t,Pp2,'-',linewidth = 3, color = clr6,label = 'Pp(t): Hipertenso')
plt.grid(False)
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4);plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t [s]', fontsize = 11)
plt.ylabel('Pp(t) [V]', fontsize = 11)
plt.legend(bbox_to_anchor = (0.5,-0.3),loc = 'center', ncol = 3,
           fontsize = 9, frameon = True)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('Sistema Cardiovascular python HipotensoPI.png',dpi=600, bbox_inches ='tight')
fg2.savefig('Sistema Cardiovascular python HipotensoPI.pdf')

fg3 = plt.figure()
plt.plot(t,Pp0,'-',linewidth = 1, color = clr1,label = 'Pp(t): Normotenso')
plt.plot(t,Pp1,'-',linewidth = 1, color = clr2,label = 'Pp(t): Hipotenso')
plt.plot(t,Pp4,'--',linewidth = 3, color = clr6,label = 'Pp(t): HipertensoPID')
plt.grid(False)
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4);plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t [s]', fontsize = 11)
plt.ylabel('Pp(t) [V]', fontsize = 11)
plt.legend(bbox_to_anchor = (0.5,-0.3),loc = 'center', ncol = 3,
           fontsize = 9, frameon = True)
plt.show()
fg3.set_size_inches(w,h)
fg3.tight_layout()
fg3.savefig('Sistema Cardiovascular python HipertensoPI.png',dpi=600, bbox_inches ='tight')
fg3.savefig('Sistema Cardiovascular python HipertensoPI.pdf')
