from det3d.datasets.cruw import CRUW3DEvaluator
import json 
import pickle 

with open('/mnt/nas_cruw/data/Day_Night_all.json', 'r') as f:
    gt_labels = json.load(f)['train']




with open('/home/andy/ipl/SMOKE/predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)

evaluator = CRUW3DEvaluator(gt_labels,[0.0, 0.1, 0.3, 0.5, 0.7])
evaluator.reset()
evaluator.process(predictions)
result = evaluator.evaluate()
for k, v in result.items():
    print(f"Evaluation {k}: {v}")