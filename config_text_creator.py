"""
Simple script to create the configuration that will be read by main.py
It saves the all in a txt file to easily change anything manually

"""

import json

K = [[641.84180665, 0., 311.13895719],
                  [0., 641.17105466, 244.65756186],
                  [0., 0., 1.]]
dist = [[-0.02331774, 0.25230237, 0., 0., -0.52186379]]

config_sp_sg = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

config_vo ={"K":K, "dist":dist, "config_sp_sg":config_sp_sg}

with open('config_vo.txt', 'w') as convert_file:
    convert_file.write(json.dumps(config_vo))

# To obtain back the data do:
with open('config_vo.txt', 'r') as convert_file:
    config_vo = json.loads(convert_file.read())