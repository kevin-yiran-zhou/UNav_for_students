from third_party.local_feature.LightGlue.lightglue import LightGlue
import loader
import torch
import yaml
from time import time


config={'width_confidence': -1,
        'depth_confidence': -1}

Model_lg=LightGlue(pretrained='superpoint', **config).eval()
local_feature_matcher = Model_lg.to('cuda')

with open('/home/unav/Desktop/UNav/configs/server.yaml', 'r') as f:
    server_config = yaml.safe_load(f)
map_data=loader.load_data(server_config)
db_name,hfile_local=map_data['database_name'],map_data['local_descriptor']

feats0=hfile_local[db_name[10]]
topk_num=40


feats1=hfile_local[db_name[1005]]

pred = {
    **{k+'0': torch.tensor(v).unsqueeze(0) for k, v in feats0.items()},
    **{k+'1': torch.tensor(v).unsqueeze(0) for k, v in feats1.items()},
}
pred = {k: v.to('cuda') for k, v in pred.items()}
pred['keypoint_scores0'] = pred.pop('scores0')
pred['keypoint_scores1'] = pred.pop('scores1')

n = 50  # Number of times to copy

keys = ['descriptors0', 'keypoints0', 'image_size0', 'keypoint_scores0',
        'descriptors1', 'keypoints1', 'image_size1', 'keypoint_scores1']

for key in keys:
    pred[key] = torch.cat([pred[key]] * n)


for i in range(10):
    last_time=time()
    # for i in range(50):
    with torch.inference_mode():
        pred1 = local_feature_matcher(pred)
        matches = pred1['matches0'].detach().cpu().short().numpy()
        print(matches.shape)
    current_time=time()
    print(current_time-last_time)

