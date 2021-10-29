"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import file_methods as fm
import pandas as pd
import sys
# import glob
import os
import shutil

def main(argv):

	try:
		directory_name=sys.argv[1]
		print(directory_name)
	except:
		print('Please pass directory_name')
		
	parsePupil(directory_name)
	print("Parsed " + directory_name + "\n")



#directory_name="F:/PROJECTS/NATURAL_DISPARITY_STAT_3/PUPIL_LABS/CALIBRATION/PUPILLABS_TEST/DATA/001"

def parsePupil(directory_name):

	pupil_filename = "pupil"
	
	if "offline" in directory_name:
		#if os.system('copy ' + directory_name + '/offline_pupil.pldata ' + directory_name + '/pupil.pldata '):
		shutil.copy2(directory_name + '/offline_pupil.pldata', directory_name + '/pupil.pldata')
		shutil.copy2(directory_name + '/offline_pupil_timestamps.npy', directory_name + '/pupil_timestamps.npy')
		
	CHECK = os.path.isfile(directory_name + "/pupil.pldata")
    
	if CHECK:
		print("FOUND FILE: " +  directory_name + "/pupil.pldata")
	else:
		print("FILE: " +  directory_name + "/pupil.pldata NOT FOUND")
	
	# data = fm.load_pldata_file(directory_name, "pupil")
	data = fm.load_pldata_file(directory_name, pupil_filename)
	DATA = pd.DataFrame([dict(d) for d in data.data])


	## TIMESTAMP
	time =  DATA.timestamp
	TIME = [t for t in time]

	## CONFIDENCE
	conf =  DATA.confidence
	CONF = [c for c in conf]

	## ELLIPSE
	ell = DATA.ellipse

	CNTR = [e['center'] for e in ell]

	X0 = [c[0] for c in CNTR]
	Y0 = [c[1] for c in CNTR]

	AXES = [e['axes'] for e in ell]
	AX = [a[0] for a in AXES]
	AY = [a[1] for a in AXES]

	THETA =  [t['angle'] for t in ell]

	## ID
	id =  DATA.id
	ID = [i for i in id]

	## SPLIT EYES
	count0 = 0
	count1 = 0

	time0 = []
	time1 = []

	conf0 = []
	conf1 = []

	x0_0 = []
	x0_1 = []
	y0_0 = []
	y0_1 = []

	ax0 = []
	ax1 = []
	ay0 = []
	ay1 = []

	theta0 = []
	theta1 = []

	for n in range(len(TIME)):
		if ID[n] == 0:
			time0.append(TIME[n])
			conf0.append(CONF[n])
			x0_0.append(X0[n])
			y0_0.append(Y0[n])
			ax0.append(AX[n])
			ay0.append(AY[n])
			theta0.append(THETA[n])
			count0 = count0 + 1
		if ID[n] == 1:
			time1.append(TIME[n])
			conf1.append(CONF[n])
			x0_1.append(X0[n])
			y0_1.append(Y0[n])
			ax1.append(AX[n])
			ay1.append(AY[n])
			theta1.append(THETA[n])
			count1 = count1 + 1


	## EYE 0 
	EYE0 = {'TIME': time0,'X0': x0_0,'Y0': y0_0,'AX': ax0,'AY': ay0,'THETA': theta0,'CONF': conf0}
	eye0 = pd.DataFrame(EYE0)
	eye0.to_csv(directory_name + '/pupil_0.csv', index = True, header=True)


	## EYE 1 
	EYE1 = {'TIME': time1,'X0': x0_1,'Y0': y0_1,'AX': ax1,'AY': ay1,'THETA': theta1,'CONF': conf1}
	eye1 = pd.DataFrame(EYE1)
	eye1.to_csv(directory_name + '/pupil_1.csv', index = True, header=True)


if __name__ == "__main__":
   main(sys.argv[1])
