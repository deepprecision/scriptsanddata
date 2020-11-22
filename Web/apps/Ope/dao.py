from apps.Ope import models

# 获取所有fuzz任务
from django.http import JsonResponse


def getallfuzz():
    data = {}
    data['rows'] = list(models.Fuzzing.objects.all().values())
    return JsonResponse(data)


def getfuzzbyid(fuzzid):
    response = models.Fuzzing.objects.get(id=fuzzid)
    return response


# 根据用户id获取fuzz任务
def getfuzzbyuid(userid):
    data = {}
    data['rows'] = list(models.Fuzzing.objects.filter(userid=userid).values())
    return JsonResponse(data)


# 根据fuzzid获取fuzz结果
def getfuzzresult(fuzzid):
    response = models.Result.objects.get(fuzzid=fuzzid)
    return response


# 根据fuzzid获取fuzz运行过程
def getfuzzruntime(fuzzid):
    data = {}
    data['rows'] = list(models.Runtime.objects.filter(fuzzid=fuzzid, iscrash=1).values())
    return JsonResponse(data)


# 获取所有seed
def getallseed(userid):
    data = {}
    data['rows'] = list(models.Seed.objects.filter(userid=userid).values())
    return data


# 根据userid删除seed
def deleteseed(userid):
    models.Seed.objects.filter(userid=userid).delete()

def handle_uploaded_file(f):
    with open('some/file/name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)