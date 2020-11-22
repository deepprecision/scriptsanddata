# models.py
from django.db import models


class Image(models.Model):
    # 图片
    img = models.ImageField(upload_to='img')
    # 创建时间
    time = models.DateTimeField(auto_now_add=True)


#
class UserInfo(models.Model):
    username = models.CharField(max_length=32)
    password = models.CharField(max_length=64)


# run.Fuzz(itf='conv1', mode=2, DLFW='pytorch', DLFW_O='tensorflow')
class Test(models.Model):
    interface = models.CharField(default='', max_length=10)
    mode = models.IntegerField(default=0)
    framework1 = models.CharField(default='', max_length=10)
    framework2 = models.CharField(default='', max_length=10)


#
class Fuzzing(models.Model):
    userid = models.IntegerField(default=0)
    interface = models.CharField(default='', max_length=15)
    mode = models.IntegerField(default=0)
    framework1 = models.CharField(default='', max_length=15)
    framework2 = models.CharField(default='', max_length=15)
    execnum = models.IntegerField(default=0)
    endcondition = models.IntegerField(default=0)
    status = models.IntegerField(default=0)
    start_time = models.CharField(default='', max_length=25)
    finish_time = models.CharField(default='', max_length=25)


class Runtime(models.Model):
    fuzzid = models.IntegerField(default=0)
    seedId = models.CharField(default='', max_length=15)
    iter = models.IntegerField(default=0)
    mutationF = models.TextField(default='')
    iscrash = models.IntegerField(default=0)
    diffVal = models.FloatField(default=0.0)
    NDArray = models.TextField(default='')
    outFram1 = models.TextField(default='')
    outFram2 = models.TextField(default='')


#
class Seed(models.Model):
    userid = models.CharField(default='', max_length=15)  # id 会自动创建,可以手动写入
    # 图片
    img = models.ImageField(upload_to='img', default='')
    # 创建时间


class Result(models.Model):
    fuzzid = models.IntegerField(default=0)
    platform = models.CharField(default='', max_length=15)
    generated = models.IntegerField(default=0)
    valid = models.IntegerField(default=0)
    crash = models.IntegerField(default=0)
    mutate_erase_bytes = models.IntegerField(default=0)
    mutate_insert_bytes = models.IntegerField(default=0)
    mutate_change_byte = models.IntegerField(default=0)
    mutate_insert_repeated_bytes = models.IntegerField(default=0)
    mutate_change_ascii_integer = models.IntegerField(default=0)
    mutate_change_bit = models.IntegerField(default=0)
    mutate_white_noise = models.IntegerField(default=0)
    mutate_rotate = models.IntegerField(default=0)
    mutate_scale = models.IntegerField(default=0)
    mutate_triangular_matrix = models.IntegerField(default=0)
    mutate_kernel_matrix = models.IntegerField(default=0)
# User(userId,name,password)//

# seed( id, encodeNDArray, owner,)

# runtime(fuzzId,iter,mutationF(int),seedId,NDArray)
# 1 0 1 0 1 0 0 0 0 0

# Fuzzing(userid,Framwork1, Framwork2,interface,status,start_time,finish_time)

# Result(fuzzing_id,platform,generated,valid,crash,v1-num,...)
