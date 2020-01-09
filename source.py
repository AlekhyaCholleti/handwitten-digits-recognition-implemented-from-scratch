import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network
net = network.Network([784,30,10],cost=network.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, test_data=test_data,
    evaluation_data=validation_data,
    lmbda = 5.0, # this is a regularization parameter
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)

