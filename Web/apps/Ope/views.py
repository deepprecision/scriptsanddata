import _thread
import time
import uuid

from apps.Ope import models
from apps.Ope.dao import *
from apps.Ope.my_forms import *
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from pandas import json
from run import Fuzz


class ImageTool:
    @staticmethod
    def get_new_random_file_name(file_name):
        find_type = False
        for c in file_name:
            if c == '.':
                find_type = True
        if find_type:
            type = file_name.split('.')[-1]
            return str(uuid.uuid1()) + '.' + type
        else:
            return str(uuid.uuid1())


@csrf_exempt
def seedupload(request):
    obj = dict()
    if request.method == 'POST':
        images = request.FILES.getlist('images')
        if images:
            for image in images:
                image.name = ImageTool.get_new_random_file_name(image.name)
                seed = models.Seed(
                    userid=1,
                    img=image
                )
                res = seed.save()
                return HttpResponse(json.dumps(res))
        else:
            obj['error'] = '没有上传的文件'
        return HttpResponse(json.dumps(obj))
    else:
        return render(request, "setting.html")

def seeddelete(request):
    userid=1
    deleteallseed(userid)
    return render(request, "setting.html", {"len": 0})



@csrf_exempt
def index(request):
    print(request.COOKIES.get('is_login'))
    status = request.COOKIES.get('is_login')  # 收到浏览器的再次请求,判断浏览器携带的cookie是不是登录成功的时候响应的 cookie
    if not status:
        return redirect('/login/')
    return render(request, "index.html")


@csrf_exempt
def login(request):
    if request.method == "GET":
        return render(request, "login.html")
    username = request.POST.get("username")
    password = request.POST.get("pwd")

    user_obj = models.UserInfo.objects.filter(username=username, password=password).first()
    print(user_obj.username)

    if not user_obj:
        return redirect("/login/")
    else:
        rep = redirect("/index/")
        rep.set_cookie("is_login", user_obj.id)
        return rep


@csrf_exempt
def register(request):
    if request.method == "GET":
        form = UserForm()  # 初始化form对象
        return render(request, "register.html", {"form": form})
    else:
        form = UserForm(request.POST)  # 将数据传给form对象
        if form.is_valid():  # 进行校验
            data = form.cleaned_data
            data.pop("r_password")
            models.UserInfo.objects.create(**data)
            return redirect("/login/")
        else:  # 校验失败
            clear_errors = form.errors.get("__all__")  # 获取全局钩子错误信息
            return render(request, "register.html", {"form": form, "clear_errors": clear_errors})


@csrf_exempt
def logout(request):
    rep = redirect('/login/')
    rep.delete_cookie("is_login")
    return rep  # 点击注销后执行,删除cookie,不再保存用户状态，并弹到登录页面


@csrf_exempt
def order(request):
    print(request.COOKIES.get('is_login'))
    status = request.COOKIES.get('is_login')
    if not status:
        return redirect('/login/')
    return render(request, "order.html")


# run.Fuzz(itf='conv1', mode=2, DLFW='pytorch', DLFW_O='tensorflow')
@csrf_exempt
def fuzz(request):
    status = request.COOKIES.get('is_login')
    if not status:
        return redirect('/login/')
    if request.method == "GET":
        form = TestForm()  # 初始化form对象
        return render(request, "fuzz.html", {"form": form})
    else:
        form = TestForm(request.POST)  # 将数据传给form对象
        if form.is_valid():  # 进行校验
            data = form.cleaned_data
            data['userid'] = status
            data['status'] = 0
            data['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            models.Fuzzing.objects.create(**data)

            response = models.Fuzzing.objects.get(userid=data.get('userid'), start_time=data.get('start_time'))
            # data.pop("r_salary")
            # models.Emp.objects.create(**data)
            # redirect("/test/")

            try:
                _thread.start_new_thread(Fuzz, (
                    '', '', data.get("interface"), data.get("mode"), data.get("framework1"), data.get("framework2"),
                    status,
                    response.id, data.get("execnum"), data.get("endcondition")))
            except:
                print("Error: 无法启动线程")

            # Fuzz(itf=data.get("interface"), mode=data.get("mode"), DLFW=data.get("framework1"),
            #          DLFW_O=data.get("framework2"), userid=status, fuzzid=response.id)
            # print(data.get("mode"))
            return redirect("/fuzz/")
        else:  # 校验失败
            clear_errors = form.errors.get("__all__")  # 获取全局钩子错误信息
            return render(request, "fuzz.html", {"form": form, "clear_errors": clear_errors})


def setting(request):
    uid = 1
    seeds = getallseed(uid)
    length = len(seeds['rows'])
    print(length)
    return render(request, "setting.html", {"len": length})


def api(request):
    return render(request, "api.html")


def fuzzing(request):
    # 执行Fuzzing任务
    # TODO
    data = {}
    data['interface'] = request.POST.get('interface', '')
    data['framework1'] = request.POST.get('framework1', '')
    data['framework2'] = request.POST.get('framework2', '')
    mode = request.POST.get('mode', '')
    if (mode == 'GPU-GPU'):
        data['mode'] = 0
    elif (mode == 'CPU-GPU'):
        data['mode'] = 1
    elif (mode == 'CPU-CPU'):
        data['mode'] = 2

    data['userid'] = 1
    data['status'] = 0
    data['execnum'] = request.POST.get('execnum', '')
    data['endcondition'] = request.POST.get('endcondition', '')
    data['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    models.Fuzzing.objects.create(**data)

    response = models.Fuzzing.objects.get(userid=data.get('userid'), start_time=data.get('start_time'))
    # data.pop("r_salary")
    # models.Emp.objects.create(**data)
    # redirect("/test/")

    try:
        _thread.start_new_thread(Fuzz, (
            '', '', data.get("interface"), data.get("mode"), data.get("framework1"), data.get("framework2"),
            0,
            response.id, data.get("execnum"), data.get("endcondition")))
    except:
        print("Error: 无法启动线程")

    # 获取任务列表
    # TODO
    return render(request, "list.html")


def result(request):
    # 获取fuzzing结果
    # TODO
    ID = request.GET.get("ID")
    print(ID)
    return render(request, "result.html", {"fuzzId": ID})


def history(request):
    return render(request, "history.html")


def list(request):
    # 获取任务列表
    # TODO
    return render(request, "list.html")


def example(request):
    fid = 12
    uid = 1
    res1 = getallfuzz()
    res2 = getfuzzbyuid(uid)
    res3 = getfuzzresult(fid)
    res4 = getfuzzruntime(fid)
    res5 = getallseed()

    print(fid)


def getFuzzList(request):
    uid = 1
    datalist = getfuzzbyuid(uid)
    # datalist = {
    #     "total": 3,
    #     "rows": [{
    #         "ID": 1,
    #         "Time": "2020-03-07 10:25",
    #         "Base": "TensorFlow",
    #         "Compare": "Caffe",
    #         "Operator": "conv",
    #         "Status": "已完成"
    #     }]
    # }
    return HttpResponse(datalist)


def getFuzzResult(request):
    fuzzId = request.GET.get("fuzzId")
    fuzz = getfuzzbyid(fuzzId)
    result = getfuzzresult(fuzzId)
    table = {
        "rows": [{
            "fuzzid": fuzz.id,
            "interface": fuzz.interface,
            "framework1": fuzz.framework1,
            "framework2": fuzz.framework2,
            "platform": fuzz.mode,
            "generated": result.generated,
            "valid": result.valid,
            "crash": result.crash
        }]
    }
    return HttpResponse(json.dumps(table))


def getChart(request):
    fuzzId = request.GET.get("fuzzId")
    result = getfuzzresult(fuzzId)
    chart = {
        "legendData": [
            'erase bytes', 'insert bytes', 'change byte', 'insert repeated bytes', 'change ascii integer', 'change bit',
            'white noise', 'rotate', 'scale', 'triangular matrix', 'kernel matrix'
        ],
        "seriesData": [
            {"value": result.mutate_erase_bytes, "name": 'erase bytes'},
            {"value": result.mutate_insert_bytes, "name": 'insert bytes'},
            {"value": result.mutate_change_byte, "name": 'change byte'},
            {"value": result.mutate_insert_repeated_bytes, "name": 'insert repeated bytes'},
            {"value": result.mutate_change_ascii_integer, "name": 'change ascii integer'},
            {"value": result.mutate_change_bit, "name": 'change bit'},
            {"value": result.mutate_white_noise, "name": 'white noise'},
            {"value": result.mutate_rotate, "name": 'rotate'},
            {"value": result.mutate_scale, "name": 'scale'},
            {"value": result.mutate_triangular_matrix, "name": 'triangular matrix'},
            {"value": result.mutate_kernel_matrix, "name": 'kernel matrix'}
        ]
    }
    return HttpResponse(json.dumps(chart))


def getCrashList(request):
    fuzzId = request.GET.get("fuzzId")
    datalist = getfuzzruntime(fuzzId)
    return HttpResponse(datalist)


def demo(request):
    return render(request, "demo.html")


def run(request):
    order = 'python run_fuzzer.py '
    # interface
    interface = request.POST.get('interface', '')
    if (interface == 'conv_same'):
        order = order + "-i conv('same') "
    elif (interface == 'conv_valid'):
        order = order + "-i conv('valid') "
    elif (interface == 'pool_max'):
        order = order + "-i pool('max') "
    elif (interface == 'pool_avg'):
        order = order + "-i pool('avg') "
    else:
        order = order + '-i ' + interface + ' '
    # base
    base = request.POST.get('base', '')
    order = order + '-base ' + base + ' '
    # compare
    compare = request.POST.get('compare', '')
    order = order + '-compare ' + compare + ' '
    # corpus
    corpus = request.POST.get('corpus', '')
    order = order + '-d ' + corpus + ' '
    # coverage
    coverage = request.POST.get('coverage', '')
    if (coverage == 'Absolute Coverage'):
        order = order + '-c absolute_coverage_function '
    elif (coverage == 'Raw Coverage'):
        order = order + '-c raw_coverage_function '
    elif (coverage == 'Neuron Coverage'):
        order = order + '-c neuron_coverage_function '
    # sampling
    sampling = request.POST.get('sampling', '')
    if (sampling == 'Uniform Sampling'):
        order = order + '-s uniform_sample_function '
    else:
        order = order + '-s recent_sample_function '
    # detection
    detection = request.POST.get('detection', '')
    if (detection == 'yes'):
        order = order + '-e difference '
    # precision
    precision = request.POST.get('precision', '')
    if (detection == "yes"):
        order = order + '-p ' + precision + ' '
    # mode
    mode = request.POST.get('mode', '')
    if (mode == 'GPU'):
        order = order + '-m 0'
    elif (mode == 'CPU-GPU'):
        order = order + '-m 1'
    elif (mode == 'CPU'):
        order = order + '-m 2'
    print(order)
    return render(request, "client.html",
                  {"order": order, "interface": interface, "base": base, "compare": compare, "corpus": corpus,
                   "coverage": coverage,
                   "sampling": sampling, "detection": detection, "precision": precision, "mode": mode})


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            # return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    return render(request, 'upfile.html', {'form': form})  # 思考一下这个return语句是否可以缩进到else语句中呢？
