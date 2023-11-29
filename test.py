import pickle
import pandas as pd

class NaiveBayesPhoneModel:
  def __init__(self, probability, binning_info):
    self.binning_info = binning_info
    self.probability = probability
    self.cols_to_drop = ['fc', 'four_g']
  
  def __convert_to_bin(self, data, binning_info):
    copied_data = data.copy()
    for column, info in binning_info.items():
        bin_edges = info['bin_edges']
        copied_data[column] = pd.cut(copied_data[column], bins=bin_edges, labels=False)

    return copied_data
  
  def predict(self, predict_data):
        # preprocess
    binned_predict_data = self.__convert_to_bin(predict_data, self.binning_info)
    binned_predict_data = binned_predict_data.drop(['price_range'] + self.cols_to_drop, axis=1)

    predictions = []

    for _, row in binned_predict_data.iterrows():
        posterior_probs = {target_class: self.probability['prior_probability'][target_class] for target_class in self.probability['prior_probability']}

        for feature_col in binned_predict_data.columns:
            for target_class in self.probability[feature_col]:
                posterior_probs[target_class] *= self.probability[feature_col][target_class][str(row[feature_col])]

        predicted_class = max(posterior_probs, key=posterior_probs.get)
        predictions.append(predicted_class)

    return predictions
  
  
# Load the serialized object from file
with open("NB_phone_model.pkl", "rb") as file:
    serialized_obj = file.read()

# Deserialize the object
obj = pickle.loads(serialized_obj)

# Access the object's attributes
print(obj.probability)  # Output: John