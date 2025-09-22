import pickle

with open(r"D:\fraud_detection\artifacts\preprocessor.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))
print(obj)
