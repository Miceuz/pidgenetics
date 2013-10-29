# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Genetic optimization of PID parameters for empyrically identified system

# <headingcell level=6>

# Albertas Mickenas mic@wemakethings.net, 2013
# copyleft, whatever

# <codecell>

#cd ~/Xaltura/pidgenetics/

# <codecell>

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import pylab as pylab
import math as math
import pid
import numpy.random.mtrand as r

def rand():
	return r.rand()

# <codecell>

#utility
def idx(x):
    return range(0, x.shape[0])

# <codecell>

#with pylab inline, make bigger images
#pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# <headingcell level=1>

# System identification

# <headingcell level=2>

# Load data and plot step response

# <codecell>

y = np.loadtxt("step-response.txt")
#normalize it
y = y - min(y)
y = y / max(y)
#plot it
plt.plot(idx(y), y, color="blue", label="Plant step response")
plt.legend()

# <headingcell level=2>

# Impulse response is a derivative of the step response

# <codecell>

#impulse response is a derivative of step response
ir = np.diff(y)

n = 33
window = signal.firwin(n, cutoff = 0.1, window = "blackman")

irFiltered = signal.lfilter(window, 1, ir)[16:]
plt.plot(idx(ir), ir, color="grey", label="Plant impulse response")
plt.plot(idx(irFiltered), irFiltered, color="red", label="filtered")
plt.legend()

# <rawcell>

# Test the impulse response by convolving it with dirac delta and compare it to our step response. 
# 
# Convolving with direac delta means applying full power to a system. 
# 
# Impulse response is everything there is to be known about the system. It fully describes the whole possible behaviour of the system (provided some some limitations - system has to be linear for this to work). To predict the behaviour of the system, for every discrete input value you have to multiply impulse response by dirac delta, scaled to input value, to get the ouptut; and then on the next input, shift previous output to the right and add new output on the top, so impulse responses overlap on top of each while time passes. This is called convolution.
# 
# Pretty good hm? Simple math at work.

# <codecell>

#dirty global output variable is born here
output = np.zeros(len(y)*2)

#convolve impuse reponse with 1. 1 * ir = y
for i in idx(y):
    for j in idx(irFiltered):
        output[i+j] = output[i+j] + ir[j]
plt.plot(idx(y), y, color="blue", label="Plant step response")
plt.plot(idx(output[:len(y)]), output[:len(y)], color="red", label="Simulated step response")

# <headingcell level=1>

# Modelling

# <headingcell level=2>

# This function processes one control step.

# <codecell>

def processStep(t, scaling):
    for j in idx(ir):
        output[t+j] = output[t+j] + ir[j] * scaling


# <headingcell level=2>

# This one evaluates how good the controller is

# <codecell>

def evaluate(p, setpoint, plot=False):
    global output
    ITAE = 0
    MSE = 0
    IAE = 0
    output = np.zeros(400 + len(ir))
    p.setPoint(setpoint)
    control = 0
    for t in range(0, 400):
        processStep(t, control)
        control = p.update(output[t])
        error = output[t] - setpoint
        ITAE = ITAE + t * abs(error)
        MSE = MSE + error**2
        IAE = IAE + abs(error)
    output = output[:-len(ir)]
    MSE = MSE / 400
    
    if plot:
        print "ITAE:", ITAE, "MSE:", MSE, "IAE", IAE
        errors = output - setpoint
        plt.figure()
        plt.plot(idx(errors), abs(errors), color="red", label="Ouput error")
        plt.title("PID: [" + str(p.Kp) + ", " + str(p.Ki) + ", " + str(p.Kd) + "] errors")
        plt.legend()
        plt.figure()
        plt.plot(idx(output), np.ones(len(output))*setpoint, color="red", label="Setpoint")    
        plt.plot(idx(output), output, color="blue", label="PID controllable model output")
        plt.legend()
        plt.title("ITAE: " + str(ITAE) + "MSE:" + str(MSE) + "IAE" + str(IAE)+ " PID: [" + str(p.Kp) + ", " + str(p.Ki) + ", " + str(p.Kd) + "]")
        
    return MSE

# <headingcell level=1>

# It's alive!

# <codecell>

POPULATION = 18
GENERATIONS = 4
MUTATION_PROBABILITY = 0.5
MUTATION_SCALE = 0.05
MAX_P = 10
MAX_I = 0.08
MAX_D = 12
population = []

# <headingcell level=2>

# Mutation and crossbreeding algorithms

# <codecell>

def mutation(maxMutation):
    if rand() < MUTATION_PROBABILITY:
        return (0.5 - rand()) * maxMutation
    else:
        return 0

def nextGeneration(population):
    population.sort(key=lambda individual: individual.getFitness())
    population = population[:-POPULATION/3*2] #part of population dies
 
    alpha = population[0]
    #alpha female mates with the best without mutation
    for i in range(0, POPULATION/3 - 1):
        mama = alpha
        papa = population[int(rand() * len(population))]
        t = rand()
        if 0 < t <= 0.33:
            P = papa.Kp# + mutation(MAX_P * MUTATION_SCALE)
            I = mama.Ki# + mutation(MAX_I * MUTATION_SCALE)
            D = mama.Kd# + mutation(MAX_D * MUTATION_SCALE)
        elif 0.33 < t <= 0.66:
            P = mama.Kp# + mutation(MAX_P * MUTATION_SCALE)
            I = papa.Ki# + mutation(MAX_I * MUTATION_SCALE)
            D = mama.Kd# + mutation(MAX_D * MUTATION_SCALE)
        else:
            P = mama.Kp# + mutation(MAX_P * MUTATION_SCALE)
            I = mama.Ki# + mutation(MAX_I * MUTATION_SCALE)
            D = papa.Kd# + mutation(MAX_D * MUTATION_SCALE)
        population.append(pid.PID(P, I, D))


    #all the other mate randomly and mutate a bit
    for i in range(0, POPULATION/3 - 1):
        papa = population[int(rand() * len(population))]
        mama = population[int(rand() * len(population))]
        t = rand()
        if 0 < t <= 0.33:
            P = papa.Kp + mutation(MAX_P * MUTATION_SCALE)
            I = mama.Ki + mutation(MAX_I * MUTATION_SCALE)
            D = mama.Kd + mutation(MAX_D * MUTATION_SCALE)
        elif 0.33 < t <= 0.66:
            P = mama.Kp + mutation(MAX_P * MUTATION_SCALE)
            I = papa.Ki + mutation(MAX_I * MUTATION_SCALE)
            D = mama.Kd + mutation(MAX_D * MUTATION_SCALE)
        else:
            P = mama.Kp + mutation(MAX_P * MUTATION_SCALE)
            I = mama.Ki + mutation(MAX_I * MUTATION_SCALE)
            D = papa.Kd + mutation(MAX_D * MUTATION_SCALE)
        population.append(pid.PID(P, I, D))
    
    #just for the kicks add a ramdom one
    population.append(pid.PID(rand() * MAX_P, rand() * MAX_I, rand() * MAX_D))
    
    #and one mutated of personally alpha
    P = alpha.Kp + mutation(MAX_P * MUTATION_SCALE)
    I = alpha.Ki + mutation(MAX_I * MUTATION_SCALE)
    D = alpha.Kd + mutation(MAX_D * MUTATION_SCALE)
    population.append(pid.PID(P, I, D))
        
    return population
    

# <headingcell level=2>

# Let ir live

# <codecell>

#utility
def evaluateFitness(population):
    print "individual evaluations of population: [",
    for individual in population:
        if -1 == individual.getFitness():
            f = evaluate(individual, 0.5)
            individual.setFitness(f)
        print individual.getFitness(),
    print "]"
    population.sort(key=lambda individual: individual.getFitness())
    

for i in range(0, POPULATION):
    population.append(pid.PID(rand() * MAX_P, rand() * MAX_I, rand() * MAX_D))

for g in range(0, GENERATIONS):
    evaluateFitness(population);
    print "Generation:", g, "Alpha individual fitness:", population[0].getFitness()
    evaluate(population[0], 0.5, True)
    population = nextGeneration(population)

# <headingcell level=3>

# A couple of winners from several runs

# <codecell>

print evaluate(pid.PID(3.645, 0.0569, -1.1012), 0.5, True)
print evaluate(pid.PID(4.0335, 0.0598, 2.6896), 0.5, True)

# <codecell>

#MSE driven

evaluate(pid.PID(14.2762504328, 0.100765749933, 11.0081466424), 0.5, True)

# <headingcell level=1>

# And now compare it with the performance of the real plant

# <codecell>

t = np.loadtxt("pid-performance.txt", delimiter=",")
t = t[:,1]
evaluate(pid.PID(4.0335, 0.0598, 2.6896), 0.5, True)
plt.plot(idx(t), t, color="green", label="Real plant")
plt.legend()
plt.figure()

