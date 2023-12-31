'''
NAME: Mohammad Saifullah Khan
ROLl NO.: 21169
EECS
ECS308
'''


from mohammad_khan_code import classification
import warnings

warnings.filterwarnings("ignore")

classifier=classification('training_data.csv', 'training_data_targets.csv', 'test_data.csv', 'rf')
classifier.classification()
predictions = classifier.predict_test_data()
