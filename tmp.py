import pickle

with open ('/home/andy/ipl/CenterRadarNet/work_dirs/kradar_4l_jde_triplet/epoch_25/test_prediction.pkl', 'rb') as f:
    pred = pickle.load(f)
    print(pred.keys())