import numpy as np

# Class definition for the Neural Network
class NeuralNetwork:
    def __init__(self, learningRate):
        self.learningRate = learningRate
        
        # Initializing weights using a normal distribution, std dev of 0.05
        self.inputWeights = np.random.normal(0.0, 0.05, (4,4))
        self.hiddenWeights = np.random.normal(0.0, 0.05, (3,4))
    
    # Implementation of the back propagation algorithm
    def backPropLearning(self,examples):
        for d in examples:
            # Formatting inputs
            inputs = np.array(d.inputArr, ndmin=2).T
            
            # Calculating the input and output of the hidden layer
            hLayerInput = np.dot(self.inputWeights,inputs)
            hLayerOutput = sigmoidFunction(hLayerInput)
            
            # Calculating the input and output of the output layer
            oLayerInput = np.dot(self.hiddenWeights, hLayerOutput)
            oLayerOutput = sigmoidFunction(oLayerInput)

            # Calculating the error for the output layer nodes and hidden layer
            # nodes
            outputError = oLayerOutput*(1-oLayerOutput)*(d.actualArr - oLayerOutput)
            hiddenError = hLayerOutput*(1-hLayerOutput)*np.dot(self.hiddenWeights.T,outputError)
            
            # Adjusting weights based on the calculated error at each node
            # (aka propagating the error backwards)
            self.hiddenWeights += self.learningRate * hLayerOutput.T * outputError
            self.inputWeights += self.learningRate * inputs.T * hiddenError
    
    # Method for validating the network (as well as calculating accuracy of
    # with given data set)
    def validate(self, examples):
        # Score arrays to be used to calculate accuracy
        expectedScore = []
        score = []
        
        # Repeated querying of the network, doesn't adjust weights of network
        for d in examples:
            # Formatting inputs
            inputs = np.array(d.inputArr, ndmin=2).T
            
            # Calculating the input and output of the hidden layer
            hLayerInput = np.dot(self.inputWeights,inputs)
            hLayerOutput = sigmoidFunction(hLayerInput)
            
            # Calculating the input and output of the output layer
            oLayerInput = np.dot(self.hiddenWeights, hLayerOutput)
            oLayerOutput = sigmoidFunction(oLayerInput)
            
            # Checking if the network guessed the correct species, updating
            # score array accordingly
            if (np.argmax(oLayerOutput) == np.argmax(d.actualArr)):
                score.append(1)
                expectedScore.append(1)
            else:
                score.append(0)
                expectedScore.append(1)
                
        # Returns the accuracy
        return ((np.asarray(score).sum()/np.asarray(score).size)*100)
    
    # Method used for querying the network with single data examples,
    # requires the data be manually inputted
    def query(self, sepalLen, sepalWid, petalLen, petalWid):
        data = [sepalLen, sepalWid, petalLen, petalWid]
        
        # Formatting inputs
        inputs = np.array(data, ndmin=2).T
        
        # Calculating the input and output of the hidden layer
        hLayerInput = np.dot(self.inputWeights,inputs)
        hLayerOutput = sigmoidFunction(hLayerInput)
        
        # Calculating the input and output of the output layer
        oLayerInput = np.dot(self.hiddenWeights, hLayerOutput)
        oLayerOutput = sigmoidFunction(oLayerInput)
        
        # Translating output values to flower species
        type = np.argmax(oLayerOutput)
        if (type == 0):
            print("Input is an Iris Setosa")
        elif (type == 1):
            print("Input is an Iris Versicolor")
        else:
            print("Input is an Iris Virginica")
    
    # Method used to generate a confusion matrix from a given data set
    def generateConfusionMatrix(self, examples):
        # This is the divisor used to report percentages, can comment out it
        # and its appearances to just report the actual number examples
        # guessed for each class
        divisor = len(examples)/3
        
        # Basic formatting for the confusion matrix, numbers here
        # represent the class number and the 1.1, 2.2, etc. represents
        # the predicted output classes (See README for more details)
        confusionMatrix = np.zeros((4,4))
        confusionMatrix[0,1] = 1
        confusionMatrix[0,2] = 2
        confusionMatrix[0,3] = 3
        confusionMatrix[1,0] = 1.1
        confusionMatrix[2,0] = 2.2
        confusionMatrix[3,0] = 3.3
        
        for d in examples:
            # Formatting inputs
            inputs = np.array(d.inputArr, ndmin=2).T
            
            # Calculating the input and output of the hidden layer
            hLayerInput = np.dot(self.inputWeights,inputs)
            hLayerOutput = sigmoidFunction(hLayerInput)
            
            # Calculating the input and output of the output layer
            oLayerInput = np.dot(self.hiddenWeights, hLayerOutput)
            oLayerOutput = sigmoidFunction(oLayerInput)
            
            # Adjusting the confusion matrix according to output values
            if (np.argmax(d.actualArr) == 0):
                if (np.argmax(oLayerOutput) == 0):
                    confusionMatrix[1,1] +=  1
                elif(np.argmax(oLayerOutput) == 1):
                    confusionMatrix[2,1] += 1
                else:
                    confusionMatrix[3,1] += 1
            elif (np.argmax(d.actualArr) == 1):
                if (np.argmax(oLayerOutput) == 0):
                    confusionMatrix[1,2] += 1
                elif(np.argmax(oLayerOutput) == 1):
                    confusionMatrix[2,2] += 1
                else:
                    confusionMatrix[3,2] += 1
            else:
                if (np.argmax(oLayerOutput) == 0):
                    confusionMatrix[1,3] += 1
                elif(np.argmax(oLayerOutput) == 1):
                    confusionMatrix[2,3] += 1
                else:
                    confusionMatrix[3,3] += 1
        
        # Comment this whole for loop out to just return the number for each
        # guessed class
        for i in range(1, 4):
            confusionMatrix[1,i] = round((confusionMatrix[1,i]/divisor*100),2)
            confusionMatrix[2,i] = round((confusionMatrix[2,i]/divisor*100),2)
            confusionMatrix[3,i] = round((confusionMatrix[3,i]/divisor*100),2)
        return confusionMatrix

# Basic class just to handle data examples more easily
class dataExample:
    def __init__(self,sepalLen,sepalWid,petalLen,petalWid,name):
        self.inputArr = [sepalLen,sepalWid,petalLen,petalWid]
        if (name == "Iris-setosa"):
            self.actualArr = [1.0,0,0]
        elif (name == "Iris-versicolor"):
            self.actualArr = [0,1.0,0]
        else:
            self.actualArr = [0,0,1.0]
        self.actualArr = np.array(self.actualArr, ndmin=2).T

# Activation funciton for the network. Could be a method of the network class,
# but not necessary with the sigmoid function implementation.
def sigmoidFunction(input):
    return 1/(1 + np.e ** -input)

# Data normalization function using the variance normalization method from
# lecture slides
def normalizeData(dataArray):
    normalizedArray = dataArray
    mean = np.mean(dataArray)
    variance = np.std(dataArray)**2
    for i in range(len(dataArray)):
        normalizedArray[i] = (dataArray[i] - mean)/variance
    return normalizedArray

# Function to normalize the entire original data set. Used to manually create
# training set, test set, validation set. I thought of coding a way to create
# the files, but the the crazy ranges in the numerous for loops required were
# too ugly for my taste. Something I could revisit in the future.
# NOTE: This function is not required if the normalized data files are provided
#       (as they will be with this submission).
def normalizeMasterSet(filename):
    # File handling
    trainingFile = open(filename, "r")
    trainingData = trainingFile.read().splitlines()
    trainingFile.close()
    
    sepalLens, sepalWids, petalLens, petalWids, classArr = [], [], [], [], []
    
    # Creating lists of each part of the data
    for d in trainingData:
        ex = d.split(',')
        sepalLens.append(ex[0])
        sepalWids.append(ex[1])
        petalLens.append(ex[2])
        petalWids.append(ex[3])
        classArr.append(ex[4])
    
    # Storing mean and variance of Sepal lengths and the normalizing the data
    sepalLens = np.asfarray(sepalLens)
    sepalLenMean = np.mean(sepalLens)
    sepalLenVar = np.std(sepalLens)**2
    sepalLens = normalizeData(sepalLens)
    
    # Storing mean and variance of Sepal widths and the normalizing the data
    sepalWids = np.asfarray(sepalWids)
    sepalWidMean = np.mean(sepalWids)
    sepalWidVar = np.std(sepalWids)**2
    sepalWids = normalizeData(sepalWids)
    
    # Storing mean and variance of Petal lengths and the normalizing the data
    petalLens = np.asfarray(petalLens)
    petalLenMean = np.mean(petalLens)
    petalLenVar = np.std(petalLens)**2
    petalLens = normalizeData(petalLens)
    
    # Storing mean and variance of Petal widths and the normalizing the data
    petalWids = np.asfarray(petalWids)
    petalWidMean = np.mean(petalWids)
    petalWidVar = np.std(petalWids)**2
    petalWids = normalizeData(petalWids)
    
    normalizedFile  = open("MasterList.txt", "w")
    
    # Writing normalized data to a new master file
    for i in range(len(sepalLens)):
        input = [str(sepalLens[i]), str(sepalWids[i]), str(petalLens[i]), str(petalWids[i]), classArr[i]]
        normalizedFile.write(" ".join(input)+"\n")
    
    normalizedFile.close()
    
    # Returning the data statistics to be used for user queries
    return [sepalLenMean,sepalLenVar,sepalWidMean,sepalWidVar,petalLenMean,petalLenVar,petalWidMean,petalWidVar]

# Given a normalized data file, this function prepares the data to be passed
# to the network.
def prepData(filename):
    # File handling
    trainingFile = open(filename, "r")
    trainingData = trainingFile.read().splitlines()
    trainingFile.close()
    
    sepalLens, sepalWids, petalLens, petalWids, classArr = [], [], [], [], []
    
    # Making lists of each part of the data
    for d in trainingData:
        ex = d.split(' ')
        sepalLens.append(ex[0])
        sepalWids.append(ex[1])
        petalLens.append(ex[2])
        petalWids.append(ex[3])
        classArr.append(ex[4])
    
    # Converting each list to an array of floats
    sepalLens = np.asfarray(sepalLens)
    sepalWids = np.asfarray(sepalWids)
    petalLens = np.asfarray(petalLens)
    petalWids = np.asfarray(petalWids)
    
    dataList = []
    
    # Turning arrays of floats into dataExample class instances and appending
    # them to a list of total data examples
    for i in range(len(sepalLens)):
        dataList.append(dataExample(sepalLens[i],sepalWids[i], petalLens[i], petalWids[i], classArr[i]))
    return dataList    

# Main driver function of the program
def main():
    # Creating master normalized file and storing data statistics
    stats = normalizeMasterSet("ANN - Iris data.txt")
    
    # Prepping ever other normalized file for input into network
    ml = prepData("MasterList.txt")
    trl = prepData("trainingData.txt")
    vl = prepData("validationData.txt")
    tl = prepData("testData.txt")
    
    # Creating an instance of the network, parameter is the learning rate
    n = NeuralNetwork(0.1)
    
    # Training the network until it reaches an accuracy above 96% with the
    # validation set
    while (n.validate(vl) < 96):
        n.backPropLearning(trl)
    
    # Printing each set's accuracy and confusion matrix
    print("Training set: Accuracy -", round(n.validate(trl), 2), "\n", n.generateConfusionMatrix(trl), "\n")
    print("Validation set: Accuracy -", round(n.validate(vl), 2), " \n", n.generateConfusionMatrix(vl), "\n")
    print("Test set: Accuracy -", round(n.validate(tl), 2), " \n", n.generateConfusionMatrix(tl), "\n")
    print("Master set: Accuracy -", round(n.validate(ml), 2), " \n", n.generateConfusionMatrix(ml), "\n")
    
    # Asking for user input
    print("Do you want to try your own input? (y/n)")
    query = input()
    if (query == 'y'):
        while (True):
            print("Input: Sepal-Length Sepal-Width Petal-Length Petal-Width")
            print("Type 'q' to quit")
            s = input()
            if (s == 'q'):
                break
            s = s.split(" ")
            s = np.asfarray(s)
            
            # Normalizing the inputted data
            s[0] = (s[0] - stats[0])/stats[1]
            s[1] = (s[1] - stats[2])/stats[3]
            s[2] = (s[2] - stats[4])/stats[5]
            s[3] = (s[3] - stats[6])/stats[7]
            
            # Querying the network with the normalized input data
            n.query(s[0],s[1],s[2],s[3])

if __name__ == '__main__':
    main()