from trainer.train import train, predict, prediction_to_class


# run predict for some text and output formatted info
def local_predict(user_input):
    # test the new model
    predictions = predict(model, user_input, v2i)
    for i in range(0, len(predictions)):
        value = predictions[i]
        class_str = prediction_to_class(i, f2i)
        print("I am {:.2f}% certain that \"".format(value * 100.0) + user_input + "\" means contact the " + class_str)
    print()


# create a graph from a series of files
# create_graph(['graph_data/news.en-00001-of-00100.parsed'], 'graph_data/graph_01.txt')

# train a neural network for intent detection
model, v2i, f2i = train('intent_training.txt', 100)

for i in range(0,5):
    print()

# test the new model
local_predict('I need to talk to the ceo !')
local_predict('can I talk to someone at your help desk please ?')

# import fileinput
# for line in fileinput.input():
#     line = line.strip()
#     if len(line) > 0:
#         local_predict(line)
#     else:
#         break
