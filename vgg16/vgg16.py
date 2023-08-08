#!/usr/bin/env python3.8
# from cProfile import label
# from logging.config import valid_ident
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
global_fontsize=14
#filter_list = {'bwaves-s.1','bwaves-s.2','xz-s.2','pop2-s.1','roms-s.1'}
#filter_list = {'bwaves-s.1','bwaves-s.2','xz-s.2'}
filter_list = {}
max_err= 11
def get_data_dict(col_key,benchmarks,data,filter=filter_list) :
    err_rate = data[col_key]
    err_rate = list(err_rate)
     
    values = err_rate
    values = [abs(value) for value in values ]
    values = values
    j=0
    value_dict={}
    for idx in range(len(benchmarks)):
        value=values[idx]
        bench = benchmarks[j]
        if(bench in filter):
            j=j+1
            continue
        value_dict[bench]=value
           
        j=j+1
    return value_dict
def get_data( file_name, sheet_name ):
    #data = pd.read_excel(open('data.xlsx','rb'),sheet_name="freq-job")
    with open(file_name,'rb') as f:
        data = pd.read_excel(f,sheet_name=sheet_name)
    
    print(data)

    benchmarks = data["benchmarks"]
    benchmarks_tmp = (list(benchmarks))

    benchmarks_raw=benchmarks_tmp
    benchmarks=[]
    for elem in benchmarks_raw:
        if type(elem) == str:
            benchmarks.append(elem)
    res = get_data_dict("abs-err",benchmarks,data)
    return res,benchmarks


file_name = "atomic-region2.xlsx"
file_name = "atomic-region-v131.xlsx"
file_name = "vgg16.xlsx"
#atomic_regions = ['looppoint','natural']
#atomic_regions = ['looppoint','mtngpp_full']
#atomic_regions = ['looppoint','looppointpp','mtng2']
#atomic_regions_name =  ['LoopPoint',"Live-LoopPoint",'OrbSim']
#atomic_regions = ['looppoint','mtng2']
#atomic_regions_name =  ['LoopPoint','Pac-Sim']
atomic_regions = ["kernellevel","wfkernel",'photon']
atomic_regions_name =  ['Kernel',"Warp+Kernel","Photon"]
dicts = []
no_flowff_value_dict,benchmarks = get_data(file_name, atomic_regions[0])
dicts.append(no_flowff_value_dict)
for i in range( 1,len(atomic_regions) ):
    flowff_value_dict,benchmarks = get_data( file_name,atomic_regions[i])
    dicts.append(flowff_value_dict)


    #plt.figure(figsize=(8, 6), dpi=80)
#plt.figure(figsize=(7, 2.75))
plt.figure(figsize=(10, 2.5))

fig=plt.subplot(1, 1, 1)
makers = ['.','^','*','x','|','1','H','S']


#dvfs_value_dict = get_data_dict("err-dvfs")
N=len(benchmarks)

width = 0.5
elem_space=2
index=np.arange(N) * elem_space
########################################
plt.rcParams['hatch.linewidth'] = 1.5
def plot_bar( data_dict, x_loc_offset ,label,color=None,hatch=None,hatchcolor=None):

    data = []
    num = 0
    for bench in benchmarks:
        if bench in filter_list:
            continue
        tmp = data_dict[bench]
        data.append(tmp)
        num += 1

    #print(max(data),min(data))
    min_data = min(data)
    max_data = max(data)

    #p2 = plt.bar(index, values, width, label="rainfall", color="#87CEFA")

    index = np.arange(num) * elem_space +width*x_loc_offset
   # plt.plot( Ks, accuracy[k_i] , label = networks[k_i] )
    print(len(index),len(data))
    p2 = fig.bar(index, data, width,label=label,hatch=hatch,edgecolor=hatchcolor,color=color)
    idx = 0
    for elem in data:
        if elem >= max_err:
            loc = max_err -2
            #fig.text( index[idx]- width * 0.38, loc, str(round(elem,2)),rotation=90,color='white',fontsize = 8,hatch=hatchs[idx])
        idx += 1
    return min_data,max_data






marker_offset = 1

#width = 0.1



#plt.xlabel('benchmarks')

plt.ylabel('Abs. Runtime Error%',fontsize = global_fontsize)


#plt.title('Err Rate of Benchmarks')


#plt.yticks(np.arange( int_min_value, int_max_value+space,space) )


min_datas=[]
max_datas=[]
idx = 0
#x_offset = [-1.5,-0.5,0.5,1.5]

atomic_region_size = len(atomic_regions)
if atomic_region_size % 2 == 1:
    x_offset = range(-atomic_region_size//2+1,atomic_region_size//2+1,1)
else:
    x_offset = np.array(range(-atomic_region_size//2,atomic_region_size//2+1,1))+0.5
print(x_offset)
x_str = []
#atomic_regions_name =  ["10M",'20M','50M','100M']
for atomic_region in atomic_regions_name:
    #if atomic_region is not "looppoint":
        #atomic_region= "atomic region = " + atomic_region 
        #atomic_region = at
     #   if 'na' not in atomic_region :
    #atomic_region +=' M'
    x_str.append( atomic_region )
colors=[ '#971e2e', '#d78763', '#7ca6cc', '#3f64a7','orange','#3fcdda','#f8eaad','#faaaae' ]
colors=[ '#d65f59','#0075b2', '#8dd3c7', '#4393c3','#7ca6cc', '#3f64a7','orange','#3fcdda','#f8eaad','#faaaae' ,'#45eff7',]
#hatchs = ["\\\\","//","xx","\\\\","++",'\\\\']
hatchs = ["","////","xx","\\\\","++",'\\\\']
colors=["#d65f59","none"]
hatchcolor=["none","#0075b2"]
colors = ['#a32a30', '#e28d69','#3264a6']
colors = ['white', "#3264a6",'#3264a6']
hatchcolors = ['#a32a30', 'none','#3264a6']
colors = [ "#3264a6","none",'#a32a30']
hatchcolors = ['#20416b','#3264a6', '#a32a30']

for dict_elem in dicts:
    x_off = x_offset[idx]
    print(x_str[idx])
    min_data,max_data = plot_bar( dict_elem, x_off , x_str[idx] ,colors[idx] , hatchs[idx],hatchcolors[idx])
    min_datas.append(min_data)
    max_datas.append(max_data)
    idx += 1
#min_data,max_data = plot_bar( no_flowff_value_dict,0.5,"w/o flowff" )
#min_datas.append(min_data)
#max_datas.append(max_data)

#fig.set_yscale("log")
min_data = min(min_datas)
max_data = max(max_datas)
#min_data = 1
#max_data = 10**( int(log10(max(max_datas)))+1)
max_data = int(max(max_datas)+1)
max_data = max_err
max_data=22
plt.ylim((0,max_data))
plt.yticks(range(0,max_data,5),fontsize=global_fontsize)


#plt.show()
#plt.legend(loc="upper right")
plt.legend(fontsize = global_fontsize,bbox_to_anchor=(0,1),loc="lower left",ncol=3,)
name = __file__ +"vgg16"

index=np.arange(len(benchmarks)) * elem_space
plt.xticks(index, benchmarks,fontsize = global_fontsize, rotation=45,ha='right', rotation_mode='anchor' )
print(benchmarks)
plt.savefig(name + ".pdf",bbox_inches='tight',dpi=300)
plt.savefig(name + ".png",bbox_inches='tight')
#plt.show()

