from semantic_graph.create_graph import create_graph
from trainer.train import train


# create a graph from a series of files
# create_graph(['graph_data/news.en-00001-of-00100.parsed'], 'graph_data/graph_01.txt')

# train a neural network for intent detection
train('intent_training.txt', 10)
