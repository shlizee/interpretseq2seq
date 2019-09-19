#human annotated segments
import os
def get_gt(data,num_trial):
    sub = {}
    if num_trial == '01':
        sub[1] = data[:550]
        sub[2] = data[550:1150]
        sub[3] = data[1150:1980]
        sub[4] = data[1980:2500]
        sub[5] = data[2500:3200]
        sub[6] = data[3200:4065]
        sub[7] = data[4065:]
        true_labels = sum([[0]*550,
                     [1]*(1150-550),
                     [0]*(1980-1150),
                     [2]*(2500-1980),
                     [0]*(3200-2500),
                     [3]*(4065-3200),
                     [2]*(4579-4065)],[])
    elif num_trial == '02':
        sub[1] = data[:1010]
        sub[2] = data[1010:1900]
        sub[3] = data[1900:2700]
        sub[4] = data[2700:3150]
        sub[5] = data[3150:4710]
        sub[6] = data[4710:5900]
        sub[7] = data[5900:7370]
        sub[8] = data[7370:8820]
        sub[9] = data[8820:9624]
        sub[10] = data[9624:]
        true_labels = sum([[0]*1010,
                     [1]*(1900-1010),
                     [2]*(2700-1900),
                     [3]*(3150-2700),
                    [4]*(4710-3150),
                     [0]*(5900-4710),
                     [5]*(7370-5900),
                     [6]*(8820-7370),
                     [7]*(9624-8820),
                     [0]*(10617-9624)],[])
    elif num_trial == '03':
        sub[1] = data[:975]
        sub[2] = data[975:1900]
        sub[3] = data[1900:2420]
        sub[4] = data[2420:3500]
        sub[5] = data[3500:4660]
        sub[6] = data[4660:5420]
        sub[7] = data[5420:6200]
        sub[8] = data[6200:7000]
        sub[9] = data[7000:]
        true_labels = sum([[0]*975,
                      [1]*(1900-975),
                      [2]*(2420-1900),
                      [0]*(3500-2420),
                      [3]*(4660-3500),
                      [4]*(5420-4660),
                      [5]*(6200-5420),
                      [6]*(7000-6200),
                      [0]*(8401-7000)],[])
    elif num_trial == '04':
        sub[1] = data[:1019]
        sub[2] = data[1019:2250]
        sub[3] = data[2250:3440]
        sub[4] = data[3440:4074]
        sub[5] = data[4074:5050]
        sub[6] = data[5050:5870]
        sub[7] = data[5870:6690]
        sub[8] = data[6690:8070]
        sub[9] = data[8070:9110]
        sub[10] = data[9110:]
        true_labels = sum([[0]*1019,
                     [1]*(2250-1019),
                     [2]*(3440-2250),
                     [3]*(4074-3440),
                     [0]*(5050-4074),
                     [4]*(5870-5050),
                     [5]*(6690-5870),
                     [6]*(8070-6690),
                     [7]*(9110-8070),
                     [0]*(10078-9110)],[])
    elif num_trial == '05':
        sub[1] = data[:818]
        sub[2] = data[818:1550]
        sub[3] = data[1550:2320]
        sub[4] = data[2320:3225]
        sub[5] = data[3225:3840]
        sub[6] = data[3840:4550]
        sub[7] = data[4550:5200]
        sub[8] = data[5200:5870]
        sub[9] = data[5870:6570]
        sub[10] = data[6570:7350]
        sub[11] = data[7350:]
        true_labels = sum([[0]*818,
                     [1]*(1550-818),
                     [2]*(2320-1550),
                     [3]*(3225-2320),
                     [4]*(3840-3225),
                     [0]*(4550-3840),
                     [5]*(5200-4550),
                     [6]*(5870-5200),
                     [7]*(6570-5870),
                     [8]*(7350-6570),
                     [0]*(8340-7350)],[])
    elif num_trial == '06':
        sub[1] = data[:1100]
        sub[2] = data[1100:1685]
        sub[3] = data[1685:2600]
        sub[4] = data[2600:3150]
        sub[5] = data[3150:3940]
        sub[6] = data[3940:4620]
        sub[7] = data[4620:5490]
        sub[8] = data[5490:6210]
        sub[9] = data[6210:7000]
        sub[10] = data[7000:8000]
        sub[11] = data[8000:8910]
        sub[12] = data[8910:]
        true_labels = sum([[0]*1100,
                     [1]*(1685-1100),
                     [2]*(2600-1685),
                     [1]*(3150-2600),
                     [3]*(3940-3150),
                     [4]*(4620-3940),
                     [5]*(5490-4620),
                     [6]*(6210-5490),
                     [7]*(7000-6210),
                     [8]*(8000-7000),
                     [9]*(8910-8000),
                     [0]*(9939-8910)],[])
    elif num_trial == '07':
        sub[1] = data[:1090]
        sub[2] = data[1090:1910]
        sub[3] = data[1910:2575]
        sub[4] = data[2575:3690]
        sub[5] = data[3690:4460]
        sub[6] = data[4460:5100]
        sub[7] = data[5100:5700]
        sub[8] = data[5700:6980]
        sub[9] = data[6980:7800]
        sub[10] = data[7800:]
        true_labels = sum([[0]*1090,
                     [1]*(1910-1090),
                     [2]*(2575-1910),
                     [1]*(3690-2575),
                     [2]*(4460-3690),
                     [3]*(5100-4460),
                     [4]*(5700-5100),
                     [0]*(6980-5700),
                     [5]*(7800-6980),
                     [0]*(8702-7800)],[])
    elif num_trial == '08':
        sub[1] = data[:1100]
        sub[2] = data[1100:1875]
        sub[3] = data[1875:2695]
        sub[4] = data[2695:3330]
        sub[5] = data[3330:3950]
        sub[6] = data[3950:4800]
        sub[7] = data[4800:5640]
        sub[8] = data[5640:6350]
        sub[9] = data[6350:7165]
        sub[10] = data[7165:8190]
        sub[11] = data[8190:]
        true_labels = sum([[0]*1100,
                     [1]*(1875-1100),
                     [2]*(2695-1875),
                     [3]*(3330-2695),
                     [4]*(3950-3330),
                     [5]*(4800-3950),
                     [6]*(5640-4800),
                     [4]*(6350-5640),
                     [7]*(7165-6350),
                     [8]*(8190-7165),
                     [0]*(9206-8190)],[])
    elif num_trial == '09':
        sub[1] = data[:1020]
        sub[2] = data[1020:1300]
        sub[3] = data[1300:2160]
        sub[4] = data[2160:2830]
        sub[5] = data[2830:3670]
        sub[6] = data[3670:]
        true_labels = sum([[0]*1020,
                     [1]*(1300-1020),
                     [2]*(2160-1300),
                     [3]*(2830-2160),
                     [4]*(3670-2830),
                     [0]*(4794-3670)],[])
    elif num_trial == '10':
        sub[1] = data[:2000]
        sub[2] = data[2000:3790]
        sub[3] = data[3790:5000]
        sub[4] = data[5000:5670]
        sub[5] = data[5670:6670]
        sub[6] = data[6670:]
        true_labels = sum([[0]*2000,
                      [1]*(3790-2000),
                      [0]*(5000-3790),
                      [2]*(5670-5000),
                      [3]*(6670-5670),
                      [0]*(7583-6670)],[])
    elif num_trial == '11':
        sub[1] = data[:1115]
        sub[2] = data[1115:1700]
        sub[3] = data[1700:2350]
        sub[4] = data[2350:2755]
        sub[5] = data[2755:4030]
        sub[6] = data[4030:4670]
        sub[7] = data[4670:]
        true_labels = sum([[0]*1115,
                      [1]*(1700-1115),
                      [2]*(2350-1700),
                      [1]*(2755-2350),
                      [2]*(4030-2755),
                      [1]*(4670-4030),
                      [0]*(5674-4670)],[])
    elif num_trial == '12':
        sub[1] = data[:1020]
        sub[2] = data[1020:1700]
        sub[3] = data[1700:3300]
        sub[4] = data[3300:4150]
        sub[5] = data[4150:5000]
        sub[6] = data[5000:5315]
        sub[7] = data[5315:7640]
        sub[8] = data[7640:]
        true_labels = sum([[0]*1020,
                       [1]*(1700-1020),
                       [2]*(3300-1700),
                       [3]*(4150-3300),
                       [4]*(5000-4150),
                       [5]*(5315-5000),
                       [6]*(7640-5315),
                       [0]*(8856-7640)],[])
    elif num_trial == '13':
        sub[1] = data[:1025]
        sub[2] = data[1025:1560]
        sub[3] = data[1560:2315]
        sub[4] = data[2315:2680]
        sub[5] = data[2680:3100]
        sub[6] = data[3100:3670]
        sub[7] = data[3670:4700]
        sub[8] = data[4700:4850]
        sub[9] = data[4850:5250]
        sub[10] = data[5250:]
        true_labels = sum([[0]*1025,
                      [1]*(1560-1025),
                      [2]*(2315-1560),
                      [3]*(2680-2315),
                      [4]*(3100-2680),
                      [1]*(3670-3100),
                      [5]*(4700-3670),
                      [6]*(4850-4700),
                      [3]*(5250-4850),
                      [0]*(6221-5250)],[])
    elif num_trial == '14':
        sub[1] = data[:680]
        sub[2] = data[680:1970]
        sub[3] = data[1970:2930]
        sub[4] = data[2930:4220]
        sub[5] = data[4220:5090]
        sub[6] = data[5090:5280]
        sub[7] = data[5280:]
        true_labels = sum([[0]*680,
                      [1]*(1970-680),
                      [2]*(2930-1970),
                      [3]*(4220-2930),
                      [2]*(5090-4220),
                      [4]*(5280-5090),
                      [5]*(6055-5280)],[])
    return sub,true_labels

def get_demo_checkpoint(demo_dir,num_trial):
    if num_trial == '01':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-9000'))
    elif num_trial == '02':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '03':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '04':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '05':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '06':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-14500'))
    elif num_trial == '07':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '08':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '09':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '10':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '11':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '12':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    elif num_trial == '13':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-9500'))
    elif num_trial == '14':
        ckpt = os.path.normpath(os.path.join(demo_dir,'25_checkpoint-10000'))
    return ckpt

