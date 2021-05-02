from fastapi import FastAPI
import uvicorn
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import numpy as np
from pydantic import BaseModel
import pickle
# Creating FastAPI instance
app = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Loading Iris Dataset
iris = load_iris()

# Getting our Features and Targets
X = iris.data
Y = iris.target

# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X, Y)
pickle.dump(clf,open('model.pkl','wb'))

# load the model from disk
loaded_model = pickle.load(open('model.pkl', 'rb'))
#@app.post('/predict')
def predict_input_page():
    st.title("ML Algorithm")
    slen=st.text_input("Sepal Length :")
    swidth=st.text_input("Sepal Width :")
    plen=st.text_input("Petal Length :")
    pwidth=st.text_input("Petal Width :")
    ok=st.button("predict the class")
    if ok:
        testdata=np.array([[slen,swidth,plen,pwidth]])
        classindx = loaded_model.predict(testdata)[0]
        st.subheader(iris.target_names[classindx])
# Creating an Endpoint to recieve the data
# to make prediction on.
@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Predicting the Class
   #class_idx = clf.predict(test_data)[0]
    class_idx = loaded_model.predict(test_data)[0]
    score=loaded_model.score(test_data)[0]
    # Return the Result
    return {'class': iris.target_names[class_idx],'score':score}


if __name__ == "__main__":
    PORT=process.env.port||3000
    app.set("port",PORT)
    uvicorn.run(app)
