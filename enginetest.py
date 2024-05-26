import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()
content = eng.load("pic.mat", nargout = 1)
data = np.array(content)