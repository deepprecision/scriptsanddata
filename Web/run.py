import os

os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
from fuzzer.new_fuzzer import Fuzzer
from utils.corpus import InputCorpus
from utils.corpus import generate_seed_corpus
from utils.counter import InitCounter
from utils.coverage_functions import neuron_coverage_origin_function
from utils.csv_writer import CSVResultWriter
from utils.decide_parameter import decide_dectection_function, decide_seed_corpus_target
from utils.mutate_functions_by_bytes import *
from utils.sample_functions import uniform_sample_function


# from utils.mutate_functions_by_bytes import mutate_batches
# from utils.mutate_functions_by_bytes import mutate_batches
# from utils.mutate_functions_by_bytes import do_basic_mutations


# Called for every client connecting (after handshake)
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    server.send_message_to_all("Hey all, a new client has joined us")


# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])


# Called when a client sends a message
def message_received(client, server, message):
    if len(message) > 200:
        message = message[:200] + '..'
    print("Client(%d) said: %s" % (client['id'], message))
    server.send_message_to_all("Client(%d) said: %s" % (client['id'], message))
    Fuzz(itf='conv1', mode=2, DLFW='tensorflow', DLFW_O='caffe')


def Fuzz(ps=None, psn='', itf='', mode=0, DLFW='caffe', DLFW_O='tensorflow', userid='', fuzzid='', execnum='',
         endcondition=''):
    # 指定待测接口
    DLFramework = DLFW
    DLFramework_Other = DLFW_O
    interface = itf
    # interface = 'pool1'
    # interface = 'pool2'
    # interface = 'conv1'
    # interface = 'relu1'
    # interface = 'sigmoid1'
    # interface = 'tanh1'
    # interface = 'dense1'

    # 指定orpus目录
    corpus_dir = Path('corpus')

    # 指定覆盖方法
    # coverage_function = absolute_coverage_function
    coverage_function = None
    coverage_name = 'None'
    # coverage_function = neuron_coverage_function
    # writer.coverage_function = 'neuron_coverage_function'

    # 指定采样方法
    sample_function = uniform_sample_function
    sample_name = 'uniform_sample_function'

    # 指定能量函数
    power_schedule = ps
    power_schedule_name = psn

    # 指定是否采用MCMC策略
    mcmcflag = 0

    # 测试精度
    precision = 8

    # 采用分析模式0:GPU;1:CPU-GPU对比模式;2:CPU
    GPU_mode = mode

    # 分析方式
    detection = decide_dectection_function(DLFramework, DLFramework_Other, interface)

    # 建立结果写入对象
    writer = CSVResultWriter(
        './cpuresults/' + DLFramework + '_' + DLFramework_Other + '_' + interface + power_schedule_name + '_' + str(
            mcmcflag) + '_' + str(GPU_mode),
        interface_name=interface, coverage_name=coverage_name, sample_name=sample_name,
        power_schedule_name=power_schedule_name, mcmc=mcmcflag, precision=precision)
    # 建立初始语料库#待修改
    seed_corpus_target = decide_seed_corpus_target(DLFramework)
    seed_corpus = generate_seed_corpus(corpus_dir=corpus_dir, target=seed_corpus_target,
                                       coverage_function=neuron_coverage_origin_function, target_interface=interface,
                                       GPU_mode=GPU_mode)
    # seed_context = {}
    # seed_context['fileid'] = corpus_dir
    # seed_context['userid'] = userid
    # seed_data = []
    # for element in seed_corpus:
    #     seed_data.append(element.data)
    # seed_context['NDArray'] = seed_data
    # if not models.Seed.objects.filter(fileid=corpus_dir):
    #     models.Seed.objects.create(**seed_context)

    input_corpus = InputCorpus(seed_corpus=seed_corpus, sample_function=sample_function,
                               coverage_function=coverage_function, threshold=0.1, algorithm='kdtree')

    mutate_data = [[None, 0, 0],
                   [mutate_erase_bytes, 12785, 24537],
                   [mutate_insert_bytes, 12980, 25702],
                   [mutate_insert_repeated_bytes, 12785, 24104],
                   [mutate_change_byte, 12879, 24552],
                   [mutate_change_bit, 12821, 24260],
                   [mutate_change_ascii_integer, 12330, 25225],
                   [mutate_white_noise, 12150, 26435],
                   [mutate_rotate, 13372, 27841],
                   [mutate_scale, 14297, 25649],
                   [mutate_triangular_matrix, 15561, 22134],
                   [mutate_kernel_matrix, 13498, 24756]]
    # 建立模糊测试器
    initcounter = InitCounter(mutate_success_counter=mutate_data)  # 参数命名需要修改

    fuzzer = Fuzzer(DLFramework=DLFramework, DLFramework_Other=DLFramework_Other, input_corpus=input_corpus,
                    coverage_function=coverage_function,
                    dectection_function=detection, initcounter=initcounter, powerschedule=power_schedule,
                    precision=precision, target_interface=interface, csvwriter=writer, mcmcflag=mcmcflag,
                    GPU_mode=GPU_mode)
    fuzzer.fuzz(fuzzid=fuzzid, seedId=corpus_dir, execnum=execnum, endcondition=endcondition)

# PORT = 9001
# server = WebsocketServer(PORT)
# server.set_fn_new_client(new_client)
# server.set_fn_client_left(client_left)
# server.set_fn_message_received(message_received)
# server.run_forever()

# if __name__ == '__main__':
#     PORT = 9001
#     server = WebsocketServer(PORT)
#     server.set_fn_new_client(new_client)
#     server.set_fn_client_left(client_left)
#     server.set_fn_message_received(message_received)
#     server.run_forever()

# Fuzz(itf='conv1', mode=2, DLFW='tensorflow', DLFW_O='caffe')
# Fuzz(itf='conv1', mode=2, DLFW='pytorch', DLFW_O='caffe')
# Fuzz(itf='conv1', mode=2, DLFW='pytorch', DLFW_O='tensorflow')
#
# Fuzz(itf='conv2', mode=2, DLFW='pytorch', DLFW_O='tensorflow')
#
# Fuzz(itf='pool1', mode=2, DLFW='tensorflow', DLFW_O='caffe')
# Fuzz(itf='pool1', mode=2, DLFW='pytorch', DLFW_O='caffe')
# Fuzz(itf='pool1', mode=2, DLFW='pytorch', DLFW_O='tensorflow')
#
# Fuzz(itf='pool2', mode=2, DLFW='tensorflow', DLFW_O='caffe')
# Fuzz(itf='pool2', mode=2, DLFW='pytorch', DLFW_O='caffe')
# Fuzz(itf='pool2', mode=2, DLFW='pytorch', DLFW_O='tensorflow')
#
# Fuzz(itf='tanh1', mode=2, DLFW='tensorflow', DLFW_O='caffe')
# Fuzz(itf='tanh1', mode=2, DLFW='pytorch', DLFW_O='caffe')
# Fuzz(itf='tanh1', mode=2, DLFW='pytorch', DLFW_O='tensorflow')
#
# Fuzz(itf='relu1', mode=2, DLFW='tensorflow', DLFW_O='caffe')
# Fuzz(itf='relu1', mode=2, DLFW='pytorch', DLFW_O='caffe')
# Fuzz(itf='relu1', mode=2, DLFW='pytorch', DLFW_O='tensorflow')
#
# Fuzz(itf='tanh1', mode=2, DLFW='tensorflow', DLFW_O='caffe')
# Fuzz(itf='tanh1', mode=2, DLFW='pytorch', DLFW_O='caffe')
# Fuzz(itf='tanh1', mode=2, DLFW='pytorch', DLFW_O='tensorflow')
#
# Fuzz(itf='sigmoid1', mode=2, DLFW='tensorflow', DLFW_O='pytorch')
# Fuzz(itf='sigmoid1', mode=2, DLFW='tensorflow', DLFW_O='pytorch')
# Fuzz(itf='sigmoid1', mode=2, DLFW='tensorflow', DLFW_O='pytorch')
