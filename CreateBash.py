output_name = 'Work.sh'
work_list = list(range(9,57,4))


with open(output_name, "w") as f:
    for w in work_list:
        #f.write('python RotateLearning.py prepare-datasets-flt -W {}\n'.format(w))
        #f.write('python RotateLearning.py train-prepared -prefix "A_" -W {} -nprep 100\n'.format(w))
        #f.write('python RotateLearning.py train-prepared -prefix "Q_" -W {} -nprep 100\n'.format(w))
        #f.write('python RotateLearning.py train-prepared -prefix "Mixed" -W {} -nprep 100\n'.format(w))
        f.write('python RotateLearning.py prepare-pmap -CB {}_Fault_Australia.hdf5 -W {}\n'.format(w,w))
        f.write('python RotateLearning.py prepare-pmap -CB {}_Fault_Quest.hdf5 -W {}\n'.format(w, w))
        f.write('python RotateLearning.py prepare-pmap -CB {}_Fault_Mixed.hdf5 -W {}\n'.format(w, w))