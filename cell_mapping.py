from numpy import *
import matplotlib.pyplot as plt

def printer():
    print('Hello cell mapping!')

ex_volts = arange(1,113)
cell_map = array([
                [57,58,59,60,3,2,1],
                [61,62,63,7,6,5,4],
                [64,65,66,67,10,9,8],
                [68,69,70,14,13,12,11],
                [71,72,73,74,17,16,15],
                [75,76,77,21,20,19,18],
                [78,79,80,81,24,23,22],
                [82,83,84,28,27,26,25],
                [85,86,87,88,31,30,29],
                [89,90,91,35,34,33,32],
                [92,93,94,95,38,37,36],
                [96,97,98,99,39,41,40],
                [100,101,102,42,43,45,44],
                [104,105,103,46,47,49,48],
                [108,107,106,50,51,53,52],
                [112,111,110,109,54,55,56]
                ])

hfdfc_cell_dict = {
 1: (0, 6),
 2: (0, 5),
 3: (0, 4),
 4: (1, 6),
 5: (1, 5),
 6: (1, 4),
 7: (1, 3),
 8: (2, 6),
 9: (2, 5),
 10: (2, 4),
 11: (3, 6),
 12: (3, 5),
 13: (3, 4),
 14: (3, 3),
 15: (4, 6),
 16: (4, 5),
 17: (4, 4),
 18: (5, 6),
 19: (5, 5),
 20: (5, 4),
 21: (5, 3),
 22: (6, 6),
 23: (6, 5),
 24: (6, 4),
 25: (7, 6),
 26: (7, 5),
 27: (7, 4),
 28: (7, 3),
 29: (8, 6),
 30: (8, 5),
 31: (8, 4),
 32: (9, 6),
 33: (9, 5),
 34: (9, 4),
 35: (9, 3),
 36: (10, 6),
 37: (10, 5),
 38: (10, 4),
 39: (11, 4),
 40: (11, 6),
 41: (11, 5),
 42: (12, 3),
 43: (12, 4),
 44: (12, 6),
 45: (12, 5),
 46: (13, 3),
 47: (13, 4),
 48: (13, 6),
 49: (13, 5),
 50: (14, 3),
 51: (14, 4),
 52: (14, 6),
 53: (14, 5),
 54: (15, 4),
 55: (15, 5),
 56: (15, 6),
 57: (0, 0),
 58: (0, 1),
 59: (0, 2),
 60: (0, 3),
 61: (1, 0),
 62: (1, 1),
 63: (1, 2),
 64: (2, 0),
 65: (2, 1),
 66: (2, 2),
 67: (2, 3),
 68: (3, 0),
 69: (3, 1),
 70: (3, 2),
 71: (4, 0),
 72: (4, 1),
 73: (4, 2),
 74: (4, 3),
 75: (5, 0),
 76: (5, 1),
 77: (5, 2),
 78: (6, 0),
 79: (6, 1),
 80: (6, 2),
 81: (6, 3),
 82: (7, 0),
 83: (7, 1),
 84: (7, 2),
 85: (8, 0),
 86: (8, 1),
 87: (8, 2),
 88: (8, 3),
 89: (9, 0),
 90: (9, 1),
 91: (9, 2),
 92: (10, 0),
 93: (10, 1),
 94: (10, 2),
 95: (10, 3),
 96: (11, 0),
 97: (11, 1),
 98: (11, 2),
 99: (11, 3),
 100: (12, 0),
 101: (12, 1),
 102: (12, 2),
 103: (13, 2),
 104: (13, 0),
 105: (13, 1),
 106: (14, 2),
 107: (14, 1),
 108: (14, 0),
 109: (15, 3),
 110: (15, 2),
 111: (15, 1),
 112: (15, 0)}

def mapToCells(arr,cell_dict = hfdfc_cell_dict):
	output = empty((16,7))
	output[:] = NaN

	if len(arr) != 112:
		print('Mismatched dimensions!')
		return output
	else:
		for ind in arange(112):
			output[cell_dict[ind + 1][0],cell_dict[ind + 1][1]] = arr[ind]
		return output

def makeCellPlot(cell_map,title = '',cunits = '',vmin = None,vmax = None,
                 merit = None,merit_label = None,merit_unit = None,save_file = None):
    fig = plt.figure(figsize = (8,8))
    extent = [0.5,7.5,16.5,0.5]
    if vmin == None:
        plt.imshow(cell_map,interpolation = 'nearest',extent = extent,aspect = 'auto')
    else:
        plt.imshow(cell_map,interpolation = 'nearest',extent = extent,aspect = 'auto',vmin = vmin, vmax = vmax)
    plt.xlabel('Column Number',fontsize = 16)
    plt.ylabel('Row Number',fontsize = 16)
    cbar = plt.colorbar()
    cbar.set_label(cunits,fontsize = 16)
    plt.title(title,fontsize = 20)

    if merit is not None:
        ax = plt.gca()
        for i in range(len(merit)):
            ax.text(0.05,0.05 + 0.05*i,merit_label[i] + ': ' + "{:4.1f}".format(merit[i]) + ' ' + merit_unit[i],
                    ha = 'left',transform = ax.transAxes, fontsize = 16)

    if save_file != None:
        plt.savefig(save_file)

    return fig
