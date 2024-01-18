import gradio as gr
from typing import List, Tuple
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd


def getPredictions(parametrs : List[float]) -> Tuple[str]:
  pred = []

  decisionTree = None
  with open("DecisionTreeClassifier.pickle", "rb") as file:
    decisionTree = pickle.load(file)

  log_reg = None
  with open("LogisticRegression.pickle", "rb") as file:
    log_reg = pickle.load(file)

  knn = None
  with open("LogisticRegression.pickle", "rb") as file:
    knn = pickle.load(file)

  pred.append(decisionTree.predict([parametrs])[0])
  pred.append(log_reg.predict([parametrs])[0])
  pred.append(knn.predict([parametrs])[0])

  #Voting prediction
  from collections import Counter
  cnt = Counter(pred)
  pred.append(cnt.most_common()[0][0])
  #
  pred = ["Мошенническая" if (x == 1) else "Немошенническая" for x in pred]

  return (pred[0], pred[1], pred[2], pred[3])


def predict(step : str, oldBalanceOrig : str, newBalanceOrig : str, oldBalanceDest : str, newBalanceDest : str, isPayment : bool) -> Tuple[str]:
    parametrs = [step, oldBalanceOrig, newBalanceOrig, oldBalanceDest, newBalanceDest, int(isPayment)]
    parametrs = [float(x) for x in parametrs]
    return getPredictions(parametrs)


def main():
  ui = gr.Interface(
      fn = predict,
      inputs = ["text", "text", "text", "text", "text",  gr.Checkbox('Is payment?')],
      outputs = [
          gr.Textbox(label="Decesion Tree prediction", lines=1),
          gr.Textbox(label="LogRegression prediction", lines=1),
          gr.Textbox(label="KNN prediction", lines=1),
          gr.Textbox(label="Voting prediction", lines=1)
      ]
  )
  ui.launch(share = True)



if __name__ == '__main__':
  main()

"""
1000
13456345
146334
134634
13463
"""