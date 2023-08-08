# from cProfile import label
# from logging.config import valid_ident
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
global_fontsize=14
global_fontsize=14+4
legend_fontsize=14
legend_space=0.4
#filter_list = {'bwaves-s.1','bwaves-s.2','xz-s.2','pop2-s.1','roms-s.1'}
#filter_list = {'bwaves-s.1','bwaves-s.2','xz-s.2'}
filter_list = {}
max_err= 11
def get_data_dict(col_key,benchmarks,data,filter=filter_list) :
    err_rate = data[col_key]
    err_rate = list(err_rate)  
     
    values = err_rate
    values = [abs(value) for value in values ]

    #j=0
    value_dict={}
    for j in range(len(benchmarks)):
        value=values[j]
        bench = benchmarks[j]
        if(bench in filter):
            #j=j+1
            continue
        value_dict[bench]=value
           
       # j=j+1
    return value_dict
def get_data( file_name, sheet_name ,column_name):
    #data = pd.read_excel(open('data.xlsx','rb'),sheet_name="freq-job")
    with open(file_name,'rb') as f:
        data = pd.read_excel(f,sheet_name=sheet_name)
    
    print(data)

    benchmarks = data["problemsize"]
    #for elem in benchmarks:
        #print(type(elem))
    benchmarks=[elem for elem in benchmarks if type(elem)==str or (not np.isnan(elem)) ]
  

    res = get_data_dict(column_name,benchmarks,data)
    return res,benchmarks


#file_name = "atomic-region2.xlsx"
#file_name = "atomic-region-v131.xlsx"
file_name_r9nano = "problemsize.xlsx"
#file_name_mi100 = "mi100problemsize.xlsx"
file_name_mi100 = "mi100.xlsx"
file_name=file_name_mi100
#atomic_regions = ['looppoint','natural']
#atomic_regions = ['looppoint','mtngpp_full']
def process_sheet_name(sheet_name,benchname,x_loc,y_loc):

    atomic_regions = [ (sheet_name,'walltime'), (sheet_name,'mixwalltime')]

    atomic_regions_name =  ['Full',"Photon"]
    dicts = []

    for sheetname,columnname in atomic_regions:
        no_flowff_value_dict,benchmarks = get_data(file_name,  sheetname,columnname)
        dicts.append(no_flowff_value_dict)
    dictsimtimes=[]

    atomic_regions = [ (sheet_name,'simtime'),(sheet_name,'mixsimtime')]
    for sheetname,columnname in atomic_regions:
        no_flowff_value_dict,benchmarks = get_data(file_name,  sheetname,columnname)
        dictsimtimes.append(no_flowff_value_dict)
    #print(dictsimtimes)

        #plt.figure(figsize=(8, 6), dpi=80)
    #plt.figure(figsize=(7, 2.75))
    #plt.figure(figsize=(10, 3))

    fig,ax1=plt.subplots(figsize=(8, 4))#1, 1, 1)
    ax2=ax1.twinx()
    makers = ['.','^','*','x','|','1','H','S']


    #dvfs_value_dict = get_data_dict("err-dvfs")
    N=len(benchmarks)

    width = 0.25
    elem_space=1
    index=np.arange(N) * elem_space
    ########################################
    def plot_bar( ax,data_dict, x_loc_offset ,label,color=None,hatch=None,hatchcolor=None):

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
        p2 = ax.bar(index, data, width,label=label,color=color,hatch=hatch,edgecolor=hatchcolor,alpha=0.6)
        idx = 0
        for elem in data:
            if elem >= max_err:
                loc = max_err -2
                #fig.text( index[idx]- width * 0.38, loc, str(round(elem,2)),rotation=90,color='white',fontsize = 8)
            idx += 1
        return data

    def plot_line( ax,data_dict, x_loc_offset ,label,color=None,hatch=None,hatchcolor=None):

        data = []
        num = 0
        for bench in benchmarks:
            if bench in filter_list:
                continue
            tmp = data_dict[bench]*1e6
            data.append(tmp)
            num += 1

        #print(max(data),min(data))
        min_data = min(data)
        max_data = max(data)

        #p2 = plt.bar(index, values, width, label="rainfall", color="#87CEFA")

        index = np.arange(num) * elem_space #+width*x_loc_offset
       # plt.plot( Ks, accuracy[k_i] , label = networks[k_i] )
        print(len(index),len(data))
        print(index,data)

        p2 = ax.plot(index, data, label=label,color=color,marker=hatch,linewidth=3,markersize=9)
        idx = 0
        for elem in data:
            if elem >= max_err:
                loc = max_err -2
                #fig.text( index[idx]- width * 0.38, loc, str(round(elem,2)),rotation=90,color='white',fontsize = 8)
            idx += 1
        return data


    # Can pass these to legend
    def combo_legend(ax1,ax2,labels):
        handler1, labeler1 = ax1.get_legend_handles_labels()
        handler2, labeler2 = ax2.get_legend_handles_labels()
        hd = []
        labli1 = list(set(labeler1))
        labli2 = list(set(labeler2))
        labli=labels
        handler=handler1+handler2
        for lab in labli:
            comb = [h for h,l in zip(handler,labeler1+labeler2) if l == lab]
            hd.append(tuple(comb))
        return hd, labli

    marker_offset = 1

    #width = 0.1



    #plt.xlabel('benchmarks')

    #plt.title('Err Rate of Benchmarks')



    plt.yticks(fontsize = global_fontsize)
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
    colors = ['#a32a30', '#e28d69','#3264a6']
    colors = ['#a32a30', '#e28d69','#3264a6']
    # hatchs = ["","////","","\\\\","++",'\\\\']
    # colors=["#d65f59","none"]

    # colors = ['#a32a30', '#e28d69','#3264a6']
    # colors = ['white', "#3264a6",'#3264a6']
    

    # colors = [ "#3264a6","none",'#a32a30']
    # hatchcolors = ['#20416b','#3264a6', '#a32a30']
    hatchs = ["","\\\\","++",'\\\\']
    colors = [ "#3264a6",'none']
    hatchcolors = ['#20416b', '#a32a30']
    full_data=[1]
    photo_data=[1]
    for idx,dict_elem in enumerate(dicts):
        x_off = x_offset[idx]
        data = plot_bar( ax2,dict_elem, x_off , x_str[idx] ,colors[idx] ,hatchs[idx],hatchcolor=hatchcolors[idx])
        if idx==0:
            full_data=data
        elif idx==len(dicts)-1:
            photo_data=data
    y_loc=max(full_data) * 0.9
    speedups=np.divide(np.array(full_data),np.array(photo_data))
    max_speedup=max(speedups)
    print("maxspeed",max_speedup)
    ax2.set_ylabel('Wall Time (s)',fontsize = global_fontsize)
    idx=0
    print(dictsimtimes)
    colors = [ "#3264a6","#a32a30",'#a32a30']
    hatchs = ["x","v","o"]
    #hatchcolors = ['#20416b','#a32a30', '#3264a6']
    full_data=[]
    photon_data=[]
    for idx,dict_elem in enumerate(dictsimtimes):
        x_off = x_offset[idx]
        print(idx)

        data = plot_line( ax1,dict_elem, x_off , x_str[idx] ,colors[idx] ,hatchs[idx],hatchcolor=hatchcolors[idx])
        if idx == 0:
            full_data= np.array(data)
        elif idx==len(dicts) -1:
            photon_data = np.array(data)
        # min_datas.append(min_data)
        # max_datas.append(max_data)
    accuracy = np.divide( np.fabs(full_data-photon_data),full_data)
    ax1.set_ylabel('Sim Time (us)',fontsize = global_fontsize)
    #plt.rc('ytick', labelsize=global_fontsize) 
    ax1.tick_params(axis='y', labelsize=global_fontsize)
    #min_data,max_data = plot_bar( no_flowff_value_dict,0.5,"w/o flowff" )
    #min_datas.append(min_data)
    #max_datas.append(max_data)

    #fig.set_yscale("log")
    #min_data = min(min_datas)
    #max_data = max(max_datas)
    #min_data = 1
    #max_data = 10**( int(log10(max(max_datas)))+1)
    #max_data = int(max(max_datas)+1)
    #max_data = max_err
    #plt.ylim((0,9))


    #plt.show()
    #plt.legend(loc="upper right")
    hd,lab=combo_legend(ax1,ax2,atomic_regions_name)
    #ax1.legend(fontsize = global_fontsize,bbox_to_anchor=(0,1),loc="lower left",ncol=len(dicts),)
    ax1.legend(hd,lab,fontsize = global_fontsize,bbox_to_anchor=(0,1),loc="lower left",ncol=len(dicts),)
    plt.text(x_loc,y_loc,r"%s (max=%.2f$\times$)"%(benchname,max_speedup),fontsize=global_fontsize*1.5)
    name = "mi100_"+sheet_name+"_accuracy"
    #benchmarks=["4M","8M","16M","32M","64M"]
    #benchmarks=[ str1.lower() for str1 in benchmarks]
    index=np.arange(len(benchmarks)) * elem_space
    benchmarks=[str(elem) for elem in benchmarks]
    ax1.set_xticks(index, benchmarks,fontsize = global_fontsize )#rotation=90,

    print(benchmarks)
    plt.savefig(name + ".pdf",bbox_inches='tight',dpi=300)
    plt.savefig(name + ".png",bbox_inches='tight')
    #plt.show()
    return list(accuracy),list(speedups)

benchnames=[     "SC",  "ReLU","MM",    "AES","SPMV","FIR"]
sheet_names=["conv","relu","matmul","aes","spmv","fir"]
text_loc=[(0,1000), (0,600),(0,5000),(0,400),(0,10000),(0,1000)]

all_accuracy=[]
all_speedup=[]
for idx in range(len(sheet_names)):
    benchname=benchnames[idx]
    sheet_name = sheet_names[idx]
    locs=text_loc[idx]
    data,speedups=process_sheet_name( sheet_name,benchname,locs[0],locs[1])
    all_accuracy+=data
    all_speedup+=speedups
average=np.average(all_accuracy)
from statistics import geometric_mean
geoaverage=geometric_mean(all_speedup)
print("average",average)
print("speedup avg",geoaverage)