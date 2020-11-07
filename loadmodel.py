#predicting for new datasets
#predicting for new datasets
from keras.preprocessing import text
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence

#this has to be loaded for new text conversion into vectors

#text='Hi I have ordered two medium pizza and two numbers of chicken wings. I have made online payment through my debit card. My order No. is 232 dated 27.06.2018 amounting to Rs. 1432. Payment has been done and then I received a call from protected telling me that chicken wings are out of stock. The payment for the same will be send back in 7 to 8 days.The main issue is that f it was not in stock then why it was not displayed. Now after payment it is being said that it is out of stock. Will you let me know how would you compensate for my grievance all My friends and Me are totally frustrated'


def load_model():
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	#load weights into new model
	loaded_model.load_weights("model.h5")

	#compile and evaluate loaded model
	loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

	tokenizer = Tokenizer(num_words=2500, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ')
	return loaded_model

def predict(text,loaded_model):
	word_index = imdb.get_word_index()

	tokenizedText= text_to_word_sequence(text)
	numericText = np.array([word_index[word] if (word in word_index) and (word_index[word]<25000) else 0 for word in tokenizedText])
	numeric_inp = sequence.pad_sequences([numericText],maxlen=70)
	num_predict = loaded_model.predict(numeric_inp)
	#return num_predict
	for item in num_predict[0]:
		if item>=0.5:
			gru_prediction='positif'
		else:
		 	gru_prediction='negatif'

	return gru_prediction

print("-----------------------------------------------")

if __name__ == '__main__':
	model=load_model()
	print()
	print()
	print()
	print()
	print()
	print()
	print("------------------------------------------")
	print()
	print()
	print()
	print("**************************************Sentiment Analysis using GRU algorithme*******************************************")
	print()
	print()
	print()
	while(1):
		print("enter your text please...")
		print()
		print(">>>",end='')
		text=input()
		gru_prediction=predict(text,model)
		print()
		print()
		print("this opinion is "+gru_prediction)

		print()
		print()
