import numpy as np
import matplotlib.pyplot as plt
import math
inf = 999999999

class AntColony:
    def __init__(self,cityCount, distances, numAnts, decay, alpha=1, beta=1):
        
        # city_count : the number of cities
        # distances : Square matrix of distances. Diagonal is assumed to be np.inf.
        # numAnts : Number of ants to find path per iteration
        # decay : the rate of decaying of pheremones (previous pheremone ) (evaporation constant)
        # alpha : exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
        # beta : exponent on distance, higher beta give distance more weight. Default=1
        
        self.cityCount = cityCount
        self.distances  = distances
        self.pheromone = np.ones((cityCount,cityCount)) / cityCount
        
        self.all_inds = range(cityCount)
        self.numAnts = numAnts
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        # to find best path and its best distance
        self.bestPath = []
        self.bestDistance = inf
        # to create generation's best values line chart
        self.currentBest = inf
        self.generationBests = []
        self.generationAvgs = []

    def run(self, n_iterations):
        # run all ants by n times to find path
        for i in range(n_iterations):
            allPaths = self.createAllAntsPaths()
            self.reCalcPheremones(allPaths)
            #self.pheromone = self.pheromone * self.decay
    
    # calculates the pheremone values with given paths
    def reCalcPheremones(self, allPaths):
        # create new matrix for calculating new pheremones
        pheromoneNew = np.zeros((self.cityCount,self.cityCount))
        # for all paths increase the pheremone in the path
        for path in allPaths:
            pathDistance = self.calPathDistance(path)
            for i in range(1, len(path)):
                # pheremone_ij += 1 / path 
                increase = 1 / pathDistance
                
                pheromoneNew[path[i - 1], path[i]] += increase
        #decay the previous pheremones according to decay parameter 
        self.pheromone = (1 - self.decay) * self.pheromone + pheromoneNew

    # creates paths for all ants
    def createAllAntsPaths(self):
        allPaths = []
        self.currentBest = inf
        generationAvg = 0
        for i in range(self.numAnts):
            path = self.createOneAntPath()
            allPaths.append(path)
            currentDis = self.calPathDistance(path)
            #try to find best path while ants travelling
            if(currentDis < self.bestDistance):
                self.bestDistance = currentDis
                self.bestPath = path
            
            # try to find best current generation path
            if(currentDis < self.currentBest):
                self.currentBest = currentDis
            generationAvg += currentDis
        self.generationAvgs.append(generationAvg/self.numAnts)
        self.generationBests.append(self.currentBest)
        
        return allPaths
    
    # creates a path with current phremones and distances
    def createOneAntPath(self):
        c=0
        path = [0]  # Start from city 0
        visited = set()
        visited.add(0)

        for i in range(len(self.distances) - 1):
            index = path[-1]
            # create probablity vector with using distances to city and phereomones on the path
            # calculate each probalbity with given equation,
            # p_i =  ( pheremone_i ^ alpha ) * ( (1 / distance_i)^beta )
            # (to make greater impact with shorter distance instead of longer distance divide 1 with distance_i) 
            p_matrix = self.pheromone[index] ** self.alpha * ((1.0 / self.distances[index]) ** self.beta)

            # make all visited cities probability to 0
            p_matrix[list(visited)] = 0

            # to calculate probability of each index divide all of them with sum
            # p_i = p_i / p_matrix's sum of all elements
            p_matrix = p_matrix / p_matrix.sum()
            
            # choose random city with given probabilities
            current = np.random.choice(self.all_inds, p=p_matrix)

            path.append(current)
            visited.add(current)

        path.append(0)  # Return to city 0
        return path

    #calculates the pat's distance
    def calPathDistance(self, path):
        distance = 0
        for i in range(len(path)-1):
            distance += self.distances[path[i], path[i + 1]]
        return distance
    
    def plotLineGraph(self):
        plt.plot(range(len(self.generationBests)),self.generationBests, linestyle='-', color='b', label = 'Best Path')
        plt.plot(range(len(self.generationBests)), self.generationAvgs, linestyle='-', color='r', label='Avgerage Path')

        plt.title('Graph of Path Lengths over Generations')
        plt.xlabel('Generations')
        plt.ylabel('Path lengths')
        plt.legend() 
        plt.grid(True)
        plt.savefig('aco.png')  # Save the graph to a PNG file
        plt.close()  # Close the plot to free memory

def readFile(file_path):
    cityCoordinates = []
    cityNames = []
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
            cityNames.append(items[0])
            
    return cityCoordinates,cityNames

def calDistance(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

cityCoordinates,cityNames = readFile("cities.txt")
numCities = len(cityCoordinates)
distances = np.zeros((numCities, numCities))
for i in range(numCities):
        distances[i,i] = 9999999999999
        for j in range(i+1, numCities):
            distance = calDistance(cityCoordinates[i], cityCoordinates[j])
            distances[i, j] = distance
            distances[j, i] = distance  # Since the distance matrix is symmetric
            

antColony = AntColony(cityCount=numCities ,distances=distances, numAnts=40, decay= 0.5, alpha=1, beta=2)
antColony.run(n_iterations=50)


print(f"Best path :", [cityNames[x] for x in antColony.bestPath])
print(len(antColony.bestPath))
print(f"Best distance :", antColony.bestDistance)
antColony.plotLineGraph()


