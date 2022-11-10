#import pylib
import cv2
import numpy as np
from matplotlib import pyplot as plt
import subprocess



def match_with_template(scan_img, template_img, use_method_nr=1, invert_template_img=False):
    # All the 6 methods for comparison in a list
    plot_sum_vec = True
    methods_list = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
                    'cv2.TM_SQDIFF_NORMED']
    use_method_nr = 1
    method = eval(methods_list[use_method_nr])
    if invert_template_img:
        template_img = template_img.__invert__()
        #print('image inverted')
    # Apply template Matching
    matchTemplate_result = cv2.matchTemplate(img_to_scan, template_img, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchTemplate_result)
    #print('min. value: {0}, max. value {1}, min. loc {2}, max. loc {3}'.format(min_val, max_val, min_loc, max_loc))
    #print('Template Image inverted: {}'.format(invert_template_img))
    #print('Comparison method used: {}'.format(methods_list[use_method_nr]))
    matchTemplate_result_norm = np.zeros(matchTemplate_result.shape)
    cv2.normalize(matchTemplate_result, matchTemplate_result_norm, 0, 1, cv2.NORM_MINMAX)
    # take sum of coloumns
    col_sum = np.cumsum(matchTemplate_result, axis=0, dtype=float)
    col_sum_vec = col_sum[-1, :]
    col_sum_vec_diff = np.diff(col_sum_vec, 1)
    if plot_sum_vec:
        plt.figure(1)
        plt.plot(col_sum_vec)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if methods_list[use_method_nr] in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc

   ##if col_sum_vec.min() < result_img_threshold:
   #    #print('Image is considered to be matched by the template.')

   #else:
   #    top_left = max_loc

    if col_sum_vec.max() > result_img_threshold or ((col_sum_vec_diff.max() > result_img_threshold and col_sum_vec.std() < col_sum_std_limit_lo)):
        #print('Image is considered to be matched by the template.')
        return True
    else:
        return False

result_img_threshold = 103
template_img_len = 60
col_sum_std_limit_lo = 9
show_images = False
invert_template_img = False

use_single_file = True

base_path = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/'
img_ext = '.png'
# template image
template_img_name = 'template_col_def_v02'
template_img_path_file = base_path + template_img_name + img_ext
# scan image
scan_img_name = 'png/mod_2163_with_col_def'
ref_img_name = 'CdTe11_vcomp_0v6_no_col'
scan_img_path_file = base_path + scan_img_name + img_ext
ref_img_path_file = base_path + ref_img_name + img_ext

# load images
image_mode_dict = {'color' : 1, 'gray_scale' : 0, 'unchanged' : -1}
image_mode = image_mode_dict['gray_scale']


ref_img = cv2.imread(ref_img_path_file, image_mode)
#template_img = cv2.imread(template_img_path_file, image_mode)
#template_img = template_img[0:template_img_len,:]
template_img = np.zeros([template_img_len,3])
template_img[:,0] = 255
template_img[:,2] = 255
#img_to_scan = ref_img

# show loaded images
if show_images:
#    plt.imshow(img_to_scan, cmap ='gray', interpolation ='bicubic')
    plt.imshow(ref_img, cmap = 'gray', interpolation = 'bicubic')
    plt.imshow(template_img, cmap ='gray', interpolation ='bicubic')


#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image',template_img)

w, h = template_img.shape[::-1]



shell_cmd_result = subprocess.check_output(['find ' + base_path + 'data/wafer_test/' + ' -name "*.png" -print'], shell=True)
print(shell_cmd_result)

shell_cmd_result_split = str(shell_cmd_result).split('\\n')
# close shell
subprocess.call("exit 0", shell=True)


#result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules/2163/ct/chip_0_81421_180807_131811_0/Vcomp_Scan_Silicon_caldet_vcomp_0.70_1A_00_00_0_00.png'
#result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules/2136/ct/chip_5_79708_180726_074109_0/Vcomp_Scan_Silicon_caldet_vcomp_0.50_1A_00_00_2_00.png'
#result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2142/ct/chip_1_80349_180727_124946_0/Vcomp_Scan_CdTe_caldet_vcomp_0.35_2A_00_00_0_00.png'
result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2226/ct/chip_7_84610_180910_122553_0/Vcomp_Scan_Silicon_caldet_vcomp_0.80_2A_00_00_2_00.png'
#result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2169/ct/chip_1_85081_180808_133730_0/Vcomp_Scan_CdTe_caldet_vcomp_0.40_2A_00_00_0_00.png'
#result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2142/ct/chip_7_79715_180727_122729_0/Vcomp_Scan_CdTe_caldet_vcomp_0.30_2A_00_00_2_00.png'
#result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//0/ct/chip_1_79754_180720_171033_1/Vcomp_Scan_CdTe_caldet_vcomp_0.30_2A_00_00_0_00.png'
#result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2267/ct/chip_1_84830_181008_154353_0/Vcomp_Scan_CdTe_caldet_vcomp_0.35_1A_00_00_0_00.png'
result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2269/ct/chip_4_84842_181009_094732_0/Vcomp_Scan_CdTe_caldet_vcomp_0.35_2A_00_00_2_00.png'
result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2213/ct/chip_1_85156_180903_114853_0/Vcomp_Scan_CdTe_caldet_vcomp_0.35_2A_00_00_0_00.png'
result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2223/ct/chip_3_84601_180906_160529_1/Vcomp_Scan_CdTe_caldet_vcomp_0.40_2A_00_00_0_00.png'
result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2270/ct/chip_2_84931_181009_171752_0/Vcomp_Scan_CdTe_caldet_vcomp_0.35_2A_00_00_0_00.png'
result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2241/ct/chip_2_84943_180917_142719_0/Vcomp_Scan_CdTe_caldet_vcomp_0.35_2A_00_00_0_00.png'
result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/modules//2236/ct/chip_0_84959_180914_090933_0/Vcomp_Scan_Silicon_caldet_vcomp_0.80_1A_00_00_0_00.png'


#result_line = '/Users/thomas.kaltenbacher/_work/testing/uv104/test_dev/col_def/data/1744_mc_17480.png'




i = 0
if use_single_file:
    img_to_scan = cv2.imread(result_line, image_mode)
    if match_with_template(img_to_scan, template_img, use_method_nr=1, invert_template_img=invert_template_img):
        print('***Image is considered to be matched by the template: {0}, id {1}'.format(result_line, i))
        cv2.imwrite(base_path + 'results/' + '{}.png'.format(i), img_to_scan)
else:
    for result_line in shell_cmd_result_split:
        result_line_split = result_line.split("b'")
        if len(result_line_split) > 1:
            result_line = result_line_split[1]

        if result_line != "'":
            img_to_scan = cv2.imread(result_line, image_mode)
            #print('---Reading File:  {0} \n'.format(result_line))

            if match_with_template(img_to_scan, template_img, use_method_nr=1, invert_template_img=invert_template_img):
                print('***Image is considered to be matched by the template: {0}, id {1}'.format(result_line,i))
                cv2.imwrite(base_path + 'results/' + '{}.png'.format(i), img_to_scan)


        i += 1



## filter/smooth the col sum vector
#col_sum_filtered  = np.convolve(col_sum_vec, [-1,5,-1], mode='same')
#col_sum_filtered_norm = col_sum_filtered/col_sum_filtered.max()
#plt.plot(col_sum_filtered)
#plt.show()
#
#bottom_right = (top_left[0] + w, top_left[1] + h)
#
#
#res_img_thr_coord = np.where(matchTemplate_result >= result_img_threshold)
#
#cv2.rectangle(img_to_scan, top_left, bottom_right, 100, 1)
#
#plt.figure(2)
#plt.subplot(121),plt.imshow(matchTemplate_result, cmap = 'gray')
#plt.subplot(122),plt.imshow(img_to_scan, cmap = 'gray')
#plt.show()

## write images
#save_img_file_name = template_img_name + '_saved'
#save_img_file = base_path + save_img_file_name + img_ext
#cv2.imwrite(save_img_file, template_img, cmap = 'gray')
#
## Finish Script
#cv2.waitKey(0)
#cv2.destroyAllWindows()



