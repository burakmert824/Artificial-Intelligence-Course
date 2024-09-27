from random import randint
import random
import math
import matplotlib.pyplot as plt



INT_MAX = 999999999
CHAR_START = 48

class indiv:
    def __init__(self) -> None:
        self.gnome = ""
        self.fitness = 0
 
    def __lt__(self, other):
        return self.fitness < other.fitness
 
    def __gt__(self, other):
        return self.fitness > other.fitness
    
def isThere(s, ch):
    for i in range(len(s)):
        if s[i] == ch:
            return True
    return False

def crossover(gnome1, gnome2,cityCount):
    firstPoint = randint(1, cityCount - 2)
    secondPoint = randint(1, cityCount - 2)

    # To ensure the points are different values
    while firstPoint == secondPoint:
        secondPoint = random.randint(2, cityCount - 2)

    smallerPointer = min(firstPoint, secondPoint)
    largerPointer = max(firstPoint, secondPoint)
    
    child1 = gnome1[0]+gnome1[smallerPointer:largerPointer]
    
    for item in gnome2:
        if item not in child1:
            child1 += item
    child1+=gnome1[0]
    
    return child1

def mutation(gnome,cityCount):
    gnome = list(gnome)
    while True:
        #first and last should remain same
        r = randint(1, cityCount-2)
        r1 = randint(1, cityCount-2)
        if r1 != r:
            temp = gnome[r]
            gnome[r] = gnome[r1]
            gnome[r1] = temp
            break
    return ''.join(gnome)


def createGnome(cityCount):
    gnome = f"{chr(CHAR_START+0)}"
    while True:
        if len(gnome) == cityCount:
            gnome += gnome[0]
            break
        temp = randint(1, cityCount-1)
        if not isThere(gnome, chr(temp + CHAR_START)):
            gnome += chr(temp + CHAR_START)
    return gnome

def calDistance(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def calFitness(gnome,cityCoordinates):   
    f = 0
    for i in range(len(gnome) - 2):
        
        a = cityCoordinates[ord(gnome[i])-CHAR_START]
        b = cityCoordinates[ord(gnome[i+1])-CHAR_START]
        f+= calDistance(a,b)
    return f


def TravelingSalesPerson(cityCoordinates,POP_SIZE,mutationRate,numBests,MAX_FITNESS_CAL,START,cityCount):
    gen = 1
    population = []
    generationBests = []
    generationAvgs = []
    
    for i in range(POP_SIZE):
        temp = indiv()
        temp.gnome = createGnome(cityCount)
        temp.fitness = calFitness(temp.gnome,cityCoordinates)
        population.append(temp)
    
    print("\n Initial Population and fitnesses ")
    for i in range(POP_SIZE):
        print(population[i].gnome,population[i].fitness)
    print()
    
    generation = 1
    maxGnome = indiv()
    maxGnome.gnome = ""
    maxGnome.fitness = INT_MAX
    
    
    while((generation * POP_SIZE < MAX_FITNESS_CAL )):
        population.sort()
        curMax = population[0]
        
        generationBests.append(curMax.fitness)
        avgFitness = 0
        for i in population:
            avgFitness += i.fitness
        generationAvgs.append(avgFitness/POP_SIZE)
        
        
        
        newPopulation = population[:numBests]
        
        for i in range(numBests,POP_SIZE):
            
            parent1 = randint(0,numBests)
            parent2 = randint(0,POP_SIZE-1)
            while(parent1 != parent2):
                parent2 = randint(0,POP_SIZE-1)
                
            childGnome = crossover(population[parent1].gnome,population[parent2].gnome,cityCount)
            
            if(randint(0,100) < mutationRate):
                childGnome = mutation(childGnome,cityCount)
            
            childIndiv = indiv()
            childIndiv.gnome = childGnome
            childIndiv.fitness = calFitness(childGnome,cityCoordinates)
            newPopulation.append(childIndiv)
                        
        population = newPopulation
        #print("Generation",gen)
        #print("GNOME FITNESS VALUE")
        
        for i in range(POP_SIZE):
            if(population[i].fitness < maxGnome.fitness):
                maxGnome = population[i]            
            #print(population[i].gnome,population[i].fitness)
        gen += 1
        generation+=1
    #for last population
    population.sort()
    curMax = population[0]
    generationBests.append(curMax.fitness)
    avgFitness = 0
    
    for i in population:
        avgFitness += i.fitness
    generationAvgs.append(avgFitness/POP_SIZE)
    
    return maxGnome,generationBests,generationAvgs

def readFile(file_path):
    cityCoordinates = []
    counter = 0
    with open(file_path, 'r') as file:
        for line in file:
            if(counter < 6) :
                counter+=1
                continue
            if(line.strip() == "EOF"):
                break
            items = line.split(" ")
            print(line.strip())  
            cityCoordinates.append((float(items[1]),float(items[2])))
    return cityCoordinates

def gnomeToPath(maxi):
    gnome = maxi.gnome
    fitness = maxi.fitness
    print(f"Fitness : {fitness}")
    ls = []
    for i in range(len(gnome)):
        ls.append(ord(gnome[i])-CHAR_START + 1)
    print(ls)
    

def plotLineGraph(generationBests,generationAvgs):
    plt.plot(range(len(generationBests)), generationBests, linestyle='-', color='b', label = 'Best Path')
    plt.plot(range(len(generationBests)), generationAvgs, linestyle='-', color='r', label='Avgerage Path')

    plt.title('Graph of Path Lengths over Generations')
    plt.xlabel('Generations')
    plt.ylabel('Path lengths')
    plt.legend()  # Show legend with labels
    plt.grid(True)
    # Add best fitness score below the plot
    best_fitness = min(generationBests)
    plt.text(0.1, -0.1, f'Best Score: {best_fitness}', ha='center', transform=plt.gca().transAxes)

    plt.savefig('graph.png')  # Save the graph to a PNG file
    plt.close()  # Close the plot to free memory


cityCoordinates = readFile("cities.txt")
cityCount = len(cityCoordinates)
print(cityCoordinates)

POP_SIZE = 100
mutationRate = 80
numBests = 10
MAX_FITNESS_CAL = 250000#250000
START = 0

max_indiv,generationBests,generationAvgs = TravelingSalesPerson(
    cityCoordinates=cityCoordinates,
    POP_SIZE=POP_SIZE,
    mutationRate=mutationRate,
    numBests=numBests,
    MAX_FITNESS_CAL= MAX_FITNESS_CAL,
    START=START,
    cityCount= len(cityCoordinates)
    )
gnomeToPath(max_indiv)

plotLineGraph(generationBests=generationBests,generationAvgs=generationAvgs)
