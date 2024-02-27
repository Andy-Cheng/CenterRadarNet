import json

with open('/mnt/ssd1/kradar_dataset/labels/refined_v3numpoints.json') as f:
    labels = json.load(f)

with open('/mnt/ssd1/kradar_dataset/labels/refined_ipl.json') as f:
    labels2 = json.load(f)

target_seq = ['51', '52', '57', '58']


for split, split_labels in labels2.items():
    for frame_label in split_labels:
        if frame_label['seq'] in target_seq:
            labels[split].append(frame_label)

with open('/mnt/ssd1/kradar_dataset/labels/refined_allv3numpoints.json', 'w') as f:
    json.dump(labels, f, indent=2)
