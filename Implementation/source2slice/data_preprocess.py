## coding:utf-8

import pickle
import os

slice_path = './test_data/4/slices/'
label_path = './test_data/4/labels/'
vuln_lines_path = './test_data/4/vulnlines/'
folder_path = './test_data/4/'
for filename in os.listdir(slice_path):
    if filename.endswith('.txt') is False:
        continue

    filepath = os.path.join(slice_path,filename)
    f = open(filepath,'r')
    slicelists = f.read().split('------------------------------')
    f.close()
    labelpath = os.path.join(label_path, filename[:-4]+'.pkl')
    vulnlines_path = os.path.join(vuln_lines_path, filename[:-4]+'.pkl')
    if os.path.isfile(labelpath) and os.path.isfile(vulnlines_path):
        print(filename)
        f = open(labelpath,'rb')
        labellists = pickle.load(f)
        f.close()

        f = open(vulnlines_path,'rb')
        vulnlineslist = pickle.load(f)
        f.close()

        if slicelists[0] == '':
            del slicelists[0]
        if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
            del slicelists[-1]

        file_path = os.path.join(folder_path, filename)
        vuln_path = os.path.join(folder_path, filename[:-4]+'.vul')
        f = open(file_path, 'a+')
        v = open(vuln_path, 'a+')
        index = -1
        for slicelist in slicelists:
            index += 1
            sentences = slicelist.split('\n')
            if sentences[0] == '\r' or sentences[0] == '':
                del sentences[0]
            if sentences == []:
                continue
            if sentences[-1] == '':
                del sentences[-1]
            if sentences[-1] == '\r':
                del sentences[-1]
            labellist = labellists[index]
            # HF: if one vulnerability is enough to label a sample as vulnerable this needs to be changed
            #for labels in labellist:
            if 1 in labellist:
                label = 1
            else:
                label = 0

            for sentence in sentences:
                f.write(str(sentence)+'\n')
            f.write(str(label)+'\n')
            f.write('------------------------------'+'\n')

            # HF: map vulnerable line to sample
            vulnlines = vulnlineslist[index]
            v.write(str(sentences[0]) + '\n')

            for vulnline in vulnlines:
                v.write(str(vulnline) + '\n')
            v.write('------------------------------' + '\n')
        f.close()
        v.close()
    else:
        print(filename + ' (label or vulnerable file not found)')
print('\success!')
        
            
    
    
