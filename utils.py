import string
import pandas as pd

def read_ov(filename: str, header='infer'):
  return pd.read_csv(filename, sep='\t', header=header)

def read_fv(filename: str, header='infer'):
  return pd.read_csv(filename, sep='\t', header=header)

def remove_punctuation(str):
  translator = str.maketrans('', '', string.punctuation)
  return str.translate(translator)