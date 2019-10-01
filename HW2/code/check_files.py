import os
andrew_id = 'XXX'


def check_file(file):
    if os.path.isfile(file):
        return True
    else:
        print('{} not found!'.format(file))
        return False
    

if ( check_file('../'+andrew_id+'/code/BRIEF.py') and \
     check_file('../'+andrew_id+'/code/keypointDetect.py') and \
     check_file('../'+andrew_id+'/code/panoramas.py') and \
     check_file('../'+andrew_id+'/code/planarH.py') and \
     check_file('../'+andrew_id+'/code/briefRotTest.py') and \
     check_file('../'+andrew_id+'/code/ar.py') and \
     check_file('../'+andrew_id+'/results/q6_1.npy') and \
     check_file('../'+andrew_id+'/results/testPattern.npy') and \
     check_file('../'+andrew_id+'/'+andrew_id+'_hw2.pdf') ):
    print('file check passed!')
else:
    print('file check failed!')

# do not modify file names - follow the given final naming policy
# images should be be included in the report