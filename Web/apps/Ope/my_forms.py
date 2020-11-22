from apps.Ope import models
from django import forms
from django.core.exceptions import ValidationError


class UserForm(forms.Form):
    username = forms.CharField(min_length=5, label="姓名", error_messages={"required": "该字段不能为空!",
                                                                         "min_length": "用户名太短。"})
    password = forms.CharField(min_length=5, label="密码", error_messages={"required": "该字段不能为空!",
                                                                         "min_length": "密码太短。"})
    r_password = forms.CharField(min_length=5, label="再次输入密码", error_messages={"required": "该字段不能为空!",
                                                                               "min_length": "密码太短。"})

    def clean_username(self):  # 局部钩子
        val = self.cleaned_data.get("username")

        if val.isdigit():
            raise ValidationError("用户名不能是纯数字")
        elif models.UserInfo.objects.filter(username=val):
            raise ValidationError("用户名已存在！")
        else:
            return val

    def clean(self):  # 全局钩子 确认两次输入的工资是否一致。
        val = self.cleaned_data.get("password")
        r_val = self.cleaned_data.get("r_password")

        if val == r_val:
            return self.cleaned_data
        else:
            raise ValidationError("请确认密码是否一致。")


class TestForm(forms.Form):
    interface = forms.CharField(label="待测接口", error_messages={"required": "该字段不能为空!"})
    mode = forms.IntegerField(label="运行环境")
    framework1 = forms.CharField(label="深度学习框架1", error_messages={"required": "该字段不能为空!"})
    framework2 = forms.CharField(label="深度学习框架2", error_messages={"required": "该字段不能为空!"})
    execnum = forms.IntegerField(label="执行次数", error_messages={"required": "该字段不能为空!"})
    endcondition = forms.IntegerField(label="目标crashs数目", error_messages={"required": "该字段不能为空!"})


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()
