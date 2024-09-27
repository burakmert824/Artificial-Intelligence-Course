from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as pyplot

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size,learning_rate):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.learning_rate = learning_rate
        self.validation_accuracies = []
    
    #hidden layer activasion function
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)
    
    #output activasion function to normalize the outputs 
    def softmax(self,weightedSumOutput):
        return np.exp(weightedSumOutput) / np.sum(np.exp(weightedSumOutput), axis=1, keepdims=True)

    #try to predict the answer
    def forward(self, inputs):
        #calculate the hidden layer output with relu activasion function and coming value
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_layer_output = self.relu(self.hidden_layer_input)

        #caluclate the output layer output with softmax activasion function and coming value
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        predicted_output = self.softmax(self.output_layer_input)
        
        #return predicated solution
        return predicted_output


    def backward(self, inputs, targets, predicted_output):
        #calculate sample count to normalize the errors
        num_samples = inputs.shape[0]
        
        #calculate output error, output_error_deriative and gradient of the weights of the output
        output_error = predicted_output - targets
        output_error_der = output_error / num_samples
        gradient_weights_hidden_output = np.dot(self.hidden_layer_output.T, output_error_der)

        #calculate the hidden error and deriative and gradient of the weights of the hidden layer
        hidden_error_der = np.dot(output_error, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_layer_input)
        gradient_weights_input_hidden = np.dot(inputs.T, hidden_error_der) / num_samples

        #decrease each weight by found gradient value by learning rate
        self.weights_input_hidden  -=  self.learning_rate * gradient_weights_input_hidden
        self.weights_hidden_output -=  self.learning_rate * gradient_weights_hidden_output
    
    #train neural network with given inputs and calculate accuracies
    def train(self, inputs, targets, epochs, x_val=None, y_val=None):
        for epoch in range(1,epochs+1):
            predicted_output = self.forward(inputs)
            self.backward(inputs, targets,predicted_output)
            if x_val is not None:
                if epoch % 10 == 0 or epoch == 1:
                    prediction,accuracy = self.predict_and_accuracy(x_val,y_val)
                    print(f"Epoch: {epoch}/{epochs}, Accuracy: {accuracy}")
                    self.validation_accuracies.append(accuracy)
                    

    #return validation accuracies
    def get_validation_accuracies(self):
        return self.validation_accuracies
    
    
    #try to predict and give accuracy of the predictions
    def predict_and_accuracy(self,x,y):
        prediction = self.forward(x)
        predictions = np.argmax(prediction, axis=1)
        accuracy = np.sum(predictions == y) / y.shape[0]
        return predictions, accuracy

def evaluate_model_and_save_histogram(neuralNetwork, X_test, y_test, output_size, filename="class_accuracies.png"):
    # Predict the output for the test dataset
    predictions, test_accuracy = neuralNetwork.predict_and_accuracy(X_test, y_test)
    print("\nTest Accuracy: {:.2f}%".format(test_accuracy * 100))
    
    # Initialize arrays for tracking total and correct predictions per class
    total_predictions = np.zeros(output_size)
    correct_predictions = np.zeros(output_size)
    
    # Calculate correct and total predictions per class
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            correct_predictions[y_test[i]] += 1
        total_predictions[y_test[i]] += 1
    
    # Calculate accuracy per class
    class_accuracies = correct_predictions / total_predictions
    for i, accuracy in enumerate(class_accuracies):
        print(f"Accuracy for class {i}: {accuracy * 100:.2f}%")
    
    # Plot histogram of class accuracies
    pyplot.figure(figsize=(10, 6))  # Adjust the figure size as needed
    pyplot.bar(range(output_size), class_accuracies * 100, tick_label=[str(i) for i in range(output_size)])
    pyplot.xlabel('Class')
    pyplot.ylabel('Accuracy (%)')
    pyplot.title('Accuracy for each class in the test dataset')
    pyplot.ylim(0, 100)  # Set the y-axis limit to [0, 100] for better visualization
    for i, accuracy in enumerate(class_accuracies):
        pyplot.text(i, accuracy * 100, f'{accuracy * 100:.2f}%', ha='center', va='bottom')
    pyplot.savefig(filename)
    pyplot.show()
    print(f"Histogram saved as {filename}")



input_size = 784 # pixel number for image (28x28)
output_size = 10 # digits [0,9]

epochs = 100

# read data divide it into labels and informations
data = pd.read_csv('assignment5.csv')
x = data.drop('label',axis=1).values
y = data['label'].values


total_count = x.shape[0]
#test_count = total_count // 5 -> %20
#validation_count = total_count // 10 %10
#train_count = total_count - (validation_count + test_count)

train_count = (total_count * 70)//100 

X_train = x[:train_count]
y_train = y[:train_count]
X_val, X_test, y_val, y_test = train_test_split(x[train_count:], y[train_count:], test_size=0.67, random_state=42)

# normalize the inputs
# scale the values to a range between 0 and 1. 
X_train = X_train/ 255
X_val   = X_val  / 255
X_test  = X_test / 255


# convert labels to vector of probability of item
# 3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 
yVector = np.eye(output_size)[y_train]



hidden_sizes = list(range(80,141,10))
learning_rates = np.arange(0.18, 0.26, 0.01)

#best
hidden_sizes = [140]
#best
learning_rates = [0.22]

# Save best accuracy % and its hyperparameters
best_accuracy = -1
best_hidden = 0
best_learning_rate = 0
best_neuralNetwork = None

#try to find best parameters for the model
for hidden_size in hidden_sizes:
    for learning_rate in learning_rates:
        # initialize neural network
        neuralNetwork = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
        neuralNetwork.train(X_train,yVector,epochs,X_val,y_val)
        accuracies = neuralNetwork.get_validation_accuracies()
        
        pyplot.plot(range(0, len([0] + accuracies)*10, 10), [0] + accuracies)
        pyplot.title('Accuracy by epoch')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Accuracy')
        pyplot.grid()
        pyplot.savefig("Accuracies.png")
        
        
        # try to predict output
        output,accuracy = neuralNetwork.predict_and_accuracy(X_val,y_val)
        print("Accuracy : hidden_size =", hidden_size, "- learning_rate =", learning_rate, " -> ",
              (accuracy * 100), "%")
        
        # Save the hyperparameters if they give the best validation accuracy so far and save the best neural network
        if accuracy > best_accuracy:
            best_hidden = hidden_size
            best_learning_rate = learning_rate
            best_accuracy = accuracy
            best_neuralNetwork = neuralNetwork

print("\nBest accuracy: ", (best_accuracy * 100), "%")
print(f"Best parameters (hidden_size = {best_hidden}, learning_rate={best_learning_rate})")

prediction, t_accuracy = best_neuralNetwork.predict_and_accuracy(X_test,y_test)
print("\nTest Accuracy: ", (t_accuracy * 100), "%")

# Create arrays for tracking predictions
total_predictions = np.zeros(output_size)
correct_predictions = np.zeros(output_size)

for i in range(len(y_test)):
    if prediction[i] == y_test[i]:
        correct_predictions[y_test[i]] += 1
    total_predictions[y_test[i]] += 1


number_accuracies = correct_predictions / total_predictions
for i, accuracy in enumerate(number_accuracies):
    print(f"Accuracy ({i}) : {accuracy * 100} %")

evaluate_model_and_save_histogram(best_neuralNetwork, X_test, y_test, output_size)


