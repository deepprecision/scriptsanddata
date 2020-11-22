# -*- coding: utf-8 -*-
import time

import numpy as np


def difference_normal(dlfw1, dlfw2):
    return np.linalg.norm(dlfw1 - dlfw2)


def difference_general(output_DLfw1, output_DLfw2):
    output_difference = abs(output_DLfw1 - output_DLfw2)
    # print(output_difference)
    return output_difference


def check_difference_framework(number1, number2, precision):
    difference = abs(number1 - number2)
    if np.max(difference) >= 10 ** (-1 * (precision - 1)):
        return False
    return True


def check_difference(input_difference, precision):
    difference = abs(input_difference)
    if np.max(difference) >= 10 ** (-1 * (precision - 1)):
        return False
    return True


def detection_nan(crashwriter, crashes,
                  input_framework1=None, input_framework1_cpu=None, input_framework2=None, input_framework2_cpu=None,
                  output_framework1=None, output_framework1_cpu=None, output_framework2=None,
                  output_framework2_cpu=None, baseID=None):
    crash_flag = 0
    if (input_framework1 is not None and np.isnan(input_framework1.all())) or (
            input_framework1_cpu is not None and np.isnan(input_framework1_cpu.all())) or \
            (input_framework2 is not None and np.isnan(input_framework2.all())) or (
            input_framework2_cpu is not None and np.isnan(input_framework2_cpu.all())) or \
            (output_framework1 is not None and np.isnan(output_framework1.all())) or (
            output_framework1_cpu is not None and np.isnan(output_framework1_cpu.all())) or \
            (output_framework2 is not None and np.isnan(output_framework2.all())) or (
            output_framework2_cpu is not None and np.isnan(output_framework2_cpu.all())):
        print('发现非数值错误')
        crashwriter.write_to_csv(crashes + 1, 0)
        crash_flag = 1
    return crash_flag


def detection_1(crashwriter, precision, crashes, framework1_name, framework2_name,
                output_framework1_mode_difference, output_framework2_mode_difference,
                output_difference_1, output_difference_2, output_difference_3, output_difference_4, baseID):
    crash_flag = 0
    if np.max(output_framework1_mode_difference) >= 10 ** (-1 * (precision - 2)):
        crash_flag = 1
        print("发现%s数值错误" % (framework1_name))
        crashwriter.write_to_csv(crashes + 1, 1,
                                 np.max(output_framework1_mode_difference),
                                 np.mean(output_framework1_mode_difference),
                                 np.linalg.norm(output_framework1_mode_difference) / np.size(
                                     output_framework1_mode_difference),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)

    if np.linalg.norm(output_framework1_mode_difference) / np.size(output_framework1_mode_difference) >= 2 * 10 ** (
            -1 * (precision - 5)):
        crash_flag = 1
        print("发现%s数值错误" % (framework1_name))
        crashwriter.write_to_csv(crashes + 1, -1,
                                 np.max(output_framework1_mode_difference),
                                 np.mean(output_framework1_mode_difference),
                                 np.linalg.norm(output_framework1_mode_difference) / np.size(
                                     output_framework1_mode_difference),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)

    if np.max(output_framework2_mode_difference) >= 10 ** (-1 * (precision - 2)):
        crash_flag = 1
        print("发现%s数值错误" % (framework2_name))
        crashwriter.write_to_csv(crashes + 1, 2,
                                 np.max(output_framework2_mode_difference),
                                 np.mean(output_framework2_mode_difference),
                                 np.linalg.norm(output_framework2_mode_difference) / np.size(
                                     output_framework2_mode_difference),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)

    if np.linalg.norm(output_framework2_mode_difference) / np.size(
            output_framework2_mode_difference) >= 2 * 10 ** (-1 * (precision - 5)):
        crash_flag = 1
        print("发现%s数值错误" % (framework2_name))
        crashwriter.write_to_csv(crashes + 1, -2,
                                 np.max(output_framework2_mode_difference),
                                 np.mean(output_framework2_mode_difference),
                                 np.linalg.norm(output_framework2_mode_difference) / np.size(
                                     output_framework2_mode_difference),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)

    if np.max(output_difference_1) >= 10 ** (-1 * (precision - 2)):
        crash_flag = 1
        print('发现CPU/GPU数值错误')
        crashwriter.write_to_csv(crashes + 1, 3,
                                 np.max(output_difference_1),
                                 np.mean(output_difference_1),
                                 np.linalg.norm(output_difference_1) / np.size(output_difference_1),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)
    if np.linalg.norm(output_difference_1) / np.size(output_difference_1) >= 2 * 10 ** (-1 * (precision - 5)):
        crash_flag = 1
        print('发现CPU/GPU数值错误')
        crashwriter.write_to_csv(crashes + 1, -3,
                                 np.max(output_difference_1),
                                 np.mean(output_difference_1),
                                 np.linalg.norm(output_difference_1) / np.size(output_difference_1),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)

    if np.max(output_difference_2) >= 10 ** (-1 * (precision - 2)):
        crash_flag = 1
        print('发现CPU/GPU数值错误')
        crashwriter.write_to_csv(crashes + 1, 4,
                                 np.max(output_difference_2),
                                 np.mean(output_difference_2),
                                 np.linalg.norm(output_difference_2) / np.size(output_difference_2),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)
    if np.linalg.norm(output_difference_2) / np.size(output_difference_2) >= 2 * 10 ** (-1 * (precision - 5)):
        crash_flag = 1
        print('发现CPU/GPU数值错误')
        crashwriter.write_to_csv(crashes + 1, -4,
                                 np.max(output_difference_2),
                                 np.mean(output_difference_2),
                                 np.linalg.norm(output_difference_2) / np.size(output_difference_2),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)

    if np.max(output_difference_3) >= 10 ** (-1 * (precision - 2)):
        crash_flag = 1
        print('发现CPU/GPU数值错误')
        crashwriter.write_to_csv(crashes + 1, 5,
                                 np.max(output_difference_3),
                                 np.mean(output_difference_3),
                                 np.linalg.norm(output_difference_3) / np.size(output_difference_3),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)
    if np.linalg.norm(output_difference_3) / np.size(output_difference_3) >= 2 * 10 ** (-1 * (precision - 5)):
        crash_flag = 1
        print('发现CPU/GPU数值错误')
        crashwriter.write_to_csv(crashes + 1, -5,
                                 np.max(output_difference_3),
                                 np.mean(output_difference_3),
                                 np.linalg.norm(output_difference_3) / np.size(output_difference_3),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)

    if np.max(output_difference_4) >= 10 ** (-1 * (precision - 2)):
        crash_flag = 1
        print('发现CPU/GPU数值错误')
        crashwriter.write_to_csv(crashes + 1, 6,
                                 np.max(output_difference_4),
                                 np.mean(output_difference_4),
                                 np.linalg.norm(output_difference_4) / np.size(output_difference_4),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)
    if np.linalg.norm(output_difference_4) / np.size(output_difference_4) >= 2 * 10 ** (-1 * (precision - 5)):
        crash_flag = 1
        print('发现CPU/GPU数值错误')
        crashwriter.write_to_csv(crashes + 1, -6,
                                 np.max(output_difference_4),
                                 np.mean(output_difference_4),
                                 np.linalg.norm(output_difference_4) / np.size(output_difference_4),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)

    return crash_flag


def detection_2(crashwriter, precision, crashes, output_difference, baseID):
    crash_flag = 0
    if np.max(output_difference) >= 10 ** (-1 * (precision - 2)):
        crash_flag = 1
        print('发现数值错误')
        crashwriter.write_to_csv(crashes + 1, 6,
                                 np.max(output_difference),
                                 np.mean(output_difference),
                                 np.linalg.norm(output_difference) / np.size(output_difference),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)
    if np.linalg.norm(output_difference) / np.size(output_difference) >= 2 * 10 ** (-1 * (precision - 5)):
        crash_flag = 1
        print('发现数值错误')
        crashwriter.write_to_csv(crashes + 1, -6,
                                 np.max(output_difference),
                                 np.mean(output_difference),
                                 np.linalg.norm(output_difference) / np.size(output_difference),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)
    return crash_flag


def detection_3(crashwriter, precision, crashes, output_difference, baseID):
    crash_flag = 0
    if np.max(output_difference) >= 10 ** (-1 * (precision - 2)):
        crash_flag = 1
        print('发现数值错误')
        crashwriter.write_to_csv(crashes + 1, 3,
                                 np.max(output_difference),
                                 np.mean(output_difference),
                                 np.linalg.norm(output_difference) / np.size(output_difference),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)

    if np.linalg.norm(output_difference) / np.size(output_difference) >= 2 * 10 ** (-1 * (precision - 5)):
        crash_flag = 1
        print('发现数值错误')
        crashwriter.write_to_csv(crashes + 1, -3,
                                 np.max(output_difference),
                                 np.mean(output_difference),
                                 np.linalg.norm(output_difference) / np.size(output_difference),
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), baseID)
    return crash_flag
