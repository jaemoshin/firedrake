from firedrake import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

streta = open('loceta.txt').read()
listeta = [line.split(',')[0] for line in streta.splitlines()]


