## read from json format to .txt format

import json

json_path = "/mnt/hdd10tb/Users/pminhtamnb/Sat_review/annotation.json"

f = open(json_path,'rb')
datas = json.load(f)
f.close()

videos_dict = {}
for i in datas:
    file_name = i['file_name']
    boxes = i['boxes'][0]
    scores = i['scores'][0]
    video_name = file_name.split("-")[0]
    if video_name in videos_dict.keys():
        videos_dict[video_name].append([file_name,boxes,scores])
    else:
        videos_dict[video_name] = []
        videos_dict[video_name].append([file_name,boxes,scores])
# print(videos_dict)

# """
for i in videos_dict.keys():
    f = open(i+".txt","w")
    frames = videos_dict[i]
    # print(frames)
    for v in frames:
        # print(v)
        f.writelines("/mnt/hdd10tb/Users/pminhtamnb/Sat_review/" + v[0] + "\t" + str(v[2]) +"\t" + str(v[1][0])+"\t" + str(v[1][1])+"\t" + str(v[1][2])+"\t" + str(v[1][3]) + "\n" )
    f.close()
# """