## read from json format to .txt format

import json

json_path = "/mnt/hdd10tb/Users/pminhtamnb/Sat_review/annotation.json"

f = open(json_path,'rb')
datas = json.load(f)
f.close()

videos_dict = {}
for i in datas:
    file_name = i['file_name']
    boxes = i['boxes']
    ids = i['ids']
    scores = i['scores']
    video_name_root = file_name.split("-")[0]
    for j in range(len(ids)):
        video_name = video_name_root + str(ids[j])
        if video_name in videos_dict.keys():
            videos_dict[video_name].append([file_name,boxes[j],scores[j]])
        else:
            videos_dict[video_name] = []
            videos_dict[video_name].append([file_name,boxes[j],scores[j]])
# print(videos_dict)

# """
for i in videos_dict.keys():
    f = open("images/" + i+".txt","w")
    frames = videos_dict[i]
    print(i)
    for v in frames:
        # print(v)
        f.writelines("/mnt/hdd10tb/Users/pminhtamnb/Sat_review/" + v[0] + "\t" + str(v[2]) +"\t" + str(v[1][0])+"\t" + str(v[1][1])+"\t" + str(v[1][2])+"\t" + str(v[1][3]) + "\n" )
    f.close()
# """