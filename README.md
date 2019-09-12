# AtJ_model_de_haze
Dense Scene Information Estimation Network for Dehazing

requirement:
python-3.5
pytorch-1.2

the original code:https://github.com/tT0NG/AtJ-DH

which is based pytorch1.0, then you will find that the model is not compatable with pytorch1.2(AvgPool2d, divisor_override),

then , please use "trandform_model.py" to load the model.
