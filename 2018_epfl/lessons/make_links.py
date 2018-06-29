import os
import glob

flist = glob.glob('../../2018/lessons/*.html')
for f in flist:
    if ('Untitled' not in f and 'exercise' not in f
         and f[f.rfind('/')+1]=='l' and 'solution' not in f):
        cmd = 'ln -s ' + f + ' ./' + f[f.rfind('/')+1:]
        os.system(cmd)

flist = glob.glob('../../2018/lessons/*.ipynb')
for f in flist:
    if ('Untitled' not in f and 'exercise' not in f
         and f[f.rfind('/')+1]=='l' and 'solution' not in f):
        cmd = 'ln -s ' + f + ' ./' + f[f.rfind('/')+1:]
        os.system(cmd)

flist = glob.glob('../../2018/lessons/*.png')
for f in flist:
    cmd = 'ln -s ' + f + ' ./' + f[f.rfind('/')+1:]
    os.system(cmd)