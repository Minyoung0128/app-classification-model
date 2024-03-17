import os
import pandas as pd 
folder ='../datasets/as'
sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

print(sub_folders)

metadata= pd.DataFrame(columns=['File','Label'])

# folder를 순회하면서, pcap file을 packet data로 만들어주기 
i=0
for sfolder_name in sub_folders:
    file_list = os.listdir(folder+'/'+sfolder_name)
    for file_name in file_list:
        
        path = folder+'/'+ sfolder_name+'/' + file_name
        
        metadata.loc[i]=[file_name,sfolder_name]
        i+=1
        
metadata.to_csv('metadata.csv',index=False)
