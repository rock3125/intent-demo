from trainer.train import train, predict, prediction_to_class


# run predict for some text and output formatted info
def local_predict(user_input):
    # test the new model
    predictions = predict(model, user_input, v2i)
    if len(predictions) == 0:
        print("Sorry, I do not understand your query\n")
    else:
        best = -1
        best_value = 0.0
        for i in range(0, len(predictions)):
            if predictions[i] > best_value:
                best_value = predictions[i]
                best = i

        # only convincing values are picked
        if best_value < 0.75:
            best = -1

        for i in range(0, len(predictions)):
            value = predictions[i]
            class_str = prediction_to_class(i, f2i)
            if i == best:
                print("I am {:.2f}% certain that \"".format(
                    value * 100.0) + user_input + "\" means contact the " + class_str + "      <-- MY PICK")
            else:
                print("I am {:.2f}% certain that \"".format(value * 100.0) +
                      user_input + "\" means contact the " + class_str)

        if best == -1:
            print("Sorry, I don't know what you mean by that.")

        print()


# create a graph from a series of files
# create_graph(['graph_data/news.en-00001-of-00100.parsed'], 'graph_data/graph_01.txt')

# train a neural network for intent detection
model, v2i, f2i = train('intent_training.txt', 10000)

# test the new model
# local_predict('I need to talk to the ceo !')
# local_predict('can I talk to someone at your help desk please ?')

import fileinput
print()
for line in fileinput.input():
    line = line.strip()
    if len(line) > 0:
        local_predict(line)
    else:
        break
