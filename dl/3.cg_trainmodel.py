from utils import tab_printer
from capsgnn import CapsGNNTrainer
from param_parser import parameter_parser

"""
todo CapsGNN 模型训练
"""

args = None
model = CapsGNNTrainer(args)
model.fit()
model.score()
model.save_predictions()
