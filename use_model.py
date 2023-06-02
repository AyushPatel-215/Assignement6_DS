import pickle

with open('model.pickle', 'rb') as file:
    loaded_model = pickle.load(file)

input_data = input("Enter the values for prediction (comma-separated): ")
input_values = list(map(float, input_data.split(",")))
# 165349.20, 136897.80, 471784.10, 0, 1

loaded_predicated_value = loaded_model.predict([input_values])
print("Loaded Predicted Value:", loaded_predicated_value[0])