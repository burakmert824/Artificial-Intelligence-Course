import numpy as np

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
        for epoch in range(1,epochs):
            predicted_output = self.forward(inputs)
            self.backward(inputs, targets,predicted_output)
            if x_val is not None:
                if epoch % 10 == 0:
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