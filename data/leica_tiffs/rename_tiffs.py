import subprocess

prefix = 'stage9_Series006_t'
suffix = '_z000_ch00.tif'

for i in range(125):
    fname = prefix + ('%02d' % i) + suffix
    subprocess.call(['cp', fname, 'leica_tiffs/'])
