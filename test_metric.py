import os
import numpy as np
from utils.evaluator import Evaluator
from utils.common import image_read_cv2
# MSRS RoadScene TNO M3FD FMB MRI_CT MRI_PET MRI_SPECT
test_folder = '/18851096398/SFMFusion/data/MSRS/test'
result_folder = '/18851096398/SFMFusion/two_stage_2_3/MSRS_size=128_weight=1/epoch_14'
output_file = '/18851096398/SFMFusion/two_stage_2_3/MSRS_size=128_weight=1/epoch_14.txt'

metric_result = np.zeros((9))

for img_name in os.listdir(os.path.join(test_folder, "ir")):
    ir = image_read_cv2(os.path.join(test_folder, "ir", img_name), 'GRAY')
    vi = image_read_cv2(os.path.join(test_folder, "vi", img_name), 'GRAY')
    fi = image_read_cv2(os.path.join(result_folder, img_name.split('.')[0] + ".png"), 'GRAY')
    # fi = image_read_cv2(os.path.join(result_folder, img_name.split('.')[0] + ".jpg"), 'GRAY')
    # fi = image_read_cv2(os.path.join(result_folder, img_name), 'GRAY')

    metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi), Evaluator.SF(fi), Evaluator.AG(fi),
                               Evaluator.MI(fi, ir, vi), Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi),
                               Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                               ])

metric_result /= len(os.listdir(result_folder))

with open(output_file, 'w') as f:
    f.write("\t\t EN\t SD\t SF\t AG\t MI\t SCD\t VIF\t Qabf\t SSIM\n")
    f.write('epoch:' + '\t'
            + str(np.round(metric_result[0], 2)) + '\t'
            + str(np.round(metric_result[1], 2)) + '\t'
            + str(np.round(metric_result[2], 2)) + '\t'
            + str(np.round(metric_result[3], 2)) + '\t'
            + str(np.round(metric_result[4], 2)) + '\t'
            + str(np.round(metric_result[5], 2)) + '\t'
            + str(np.round(metric_result[6], 2)) + '\t'
            + str(np.round(metric_result[7], 2)) + '\t'
            + str(np.round(metric_result[8], 2)) + '\t'
            + '\n')
    f.write("=" * 80 + '\n')
