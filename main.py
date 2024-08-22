import argparse
from torch.backends import cudnn
from utils.utils import *
from solver import Solver
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import time
import GPUtil
import time
import psutil


stopped_num = 10000000     # （设置一个最大获取次数，防止记录文本爆炸）
delay = 10  # 采样信息时间间隔
Gpus = GPUtil.getGPUs()


def get_gpu_info():
    '''
    :return:
    '''
    gpulist = []
    GPUtil.showUtilization()

    # 获取多个GPU的信息，存在列表里
    for gpu in Gpus:
        print('gpu.id:', gpu.id)
        print('GPU总量：', gpu.memoryTotal)
        print('GPU使用量：', gpu.memoryUsed)
        print('gpu使用占比:', gpu.memoryUtil * 100)
        # 按GPU逐个添加信息
        gpulist.append([gpu.id, gpu.memoryTotal, gpu.memoryUsed, gpu.memoryUtil * 100])

    return gpulist


def get_cpu_info():
    ''' :return:
    memtotal: 总内存
    memfree: 空闲内存
    memused: Linux: total - free,已使用内存
    mempercent: 已使用内存占比
    cpu: 各个CPU使用占比
    '''
    mem = psutil.virtual_memory()
    memtotal = mem.total
    memfree = mem.free
    mempercent = mem.percent
    memused = mem.used
    cpu = psutil.cpu_percent(percpu=True)

    return memtotal, memfree, memused, mempercent, cpu



def main(config):

    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))
    solver.test()
    times = 0
    while True:
        # 最大循环次数
        if times < stopped_num:
            # 打印当前时间
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # 获取CPU信息
            cpu_info = get_cpu_info()
            # 获取GPU信息
            gpu_info = get_gpu_info()
            # 添加时间间隙
            print(cpu_info)
            print(gpu_info, '\n')
            time.sleep(delay)
            times += 1
        else:
            break
    return solver

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()

    # Alternative
    dataset_name='SWAT'
    parser.add_argument('--win_size', type=int, default=10) #######  10   局部窗口长度
    parser.add_argument('--win_size_1', type=int, default=20)  ####### 50  全局窗口长度
    parser.add_argument('--count', type=int, default=21)  #########30    必须为奇数
    parser.add_argument('--anormly_ratio', type=float, default=0.9)########## 5
    parser.add_argument('--p', type=float, default=0.1)  ########## 大窗口小窗口分数占比   p*点级别+(1-p)*子序列
    parser.add_argument('--select', type=float, default=1)  ########## 0表示振幅，1 表示 相位
    parser.add_argument('--input_c', type=int, default=51)  ##########
    parser.add_argument('--batch_size', type=int, default=256)####### 512

    parser.add_argument('--dataset', type=str, default=f'{dataset_name}')  #######
    parser.add_argument('--data_path', type=str, default=f'{dataset_name}')######

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

    # Default
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')



    config = parser.parse_args()
    args = vars(config)
    print(f"dataset: {config.dataset}  \n win_size: {config.win_size}, win_size_1: {config.win_size_1}" 
          f"\n  count: {config.count}, anormly_ratio:{config.anormly_ratio},input_c:{config.input_c}"
          f"\n p: {config.p}, select: {config.select} ,batch_size: {config.batch_size}"
          )
    
    if config.dataset == 'UCR':
        batch_size_buffer = [2,4,8,16,32,64,128,256]
        data_len = np.load('dataset/'+config.data_path + "/UCR_"+str(config.index)+"_train.npy").shape[0] 

    elif config.dataset == 'UCR_AUG':
        batch_size_buffer = [2,4,8,16,32,64,128,256]
        data_len = np.load('dataset/'+config.data_path + "/UCR_AUG_"+str(config.index)+"_train.npy").shape[0] 

    elif config.dataset == 'SMD_Ori':
        batch_size_buffer = [2,4,8,16,32,64,128,256,512]
        data_len = np.load('dataset/'+config.data_path + "/SMD_Ori_"+str(config.index)+"_train.npy").shape[0] 

        
    
    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ','')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]

    main(config)
    end_time = time.time()
    print("total time:", end_time-start_time)
    
