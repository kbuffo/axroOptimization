from scipy import ndimage as nd
import numpy as np
from operator import itemgetter
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
plt.rcParams['savefig.facecolor']='white'

import utilities.figure_plotting as fp
import imaging.man as man
import imaging.analysis as alsis
import axroOptimization.evaluateMirrors as eva
import axroOptimization.solver as solver

def printer():
    print('Hello cell functions!')

############### Constructing cell, pin, and cable structure ####################

N_cells = 288 # number of cells
N_rows = 18 # number of rows (axial dimension)
N_cols = 16 # number of columns (azimuthal dimension)
N_ppins = 104 # number of pins on a P cable
N_p0x_pins = 51 # number of pins on a P0X cable
N_region_pins = 40 # number of pins per region

cell_nums = np.arange(1, N_cells+1, 1) # array of cell numbers
rows = np.arange(0, N_rows, 1) # array of cell rows
cols = np.arange(0, N_cols, 1) # array of cell columns
# list of grid coordinates for each cell in the standard format
grid_coords = [[rows[i%N_rows],cols[i//N_rows]] for i in range(N_cells)]
# list of row-column index lables for each cell in the standard format
rc_labels = ['R'+str(coord[0]+1)+'C'+str(coord[1]+1) for coord in grid_coords]
# construct the list of cells in each region in the pinout format
concs = ['GND', 'AOUT-XX', 'AGND', 'NC']
la_cells = [concs[0], concs[1]] + cell_nums[0:32].tolist() + [concs[1]]*7 + [concs[2]] + [concs[3]]*9
lb_cells = [concs[2]] + cell_nums[32:72].tolist() + [concs[2]] + [concs[3]]*9
lc_cells = [concs[2]] + cell_nums[72:111].tolist() + [concs[1], concs[2]] + [concs[3]]*9
ld_cells = [concs[2]] + cell_nums[111:144].tolist() + [concs[1]]*7 + [concs[2]] + [concs[3]]*9
ra_cells = [concs[2]] + [concs[1]]*7 + np.flip(cell_nums[144:176]).tolist() + [concs[1], concs[0]] + [concs[3]]*9
rb_cells = [int(cell_nums[215])] + [concs[1]] + np.flip(cell_nums[176:215]).tolist() + [concs[2]] + [concs[3]]*9
rc_cells = [concs[2], concs[1]] + np.flip(cell_nums[216:255]).tolist() + [concs[2]] + [concs[3]]*9
rd_cells = [concs[2]] + [concs[1]]*7 + np.flip(cell_nums[255:288]).tolist() + [concs[2]] + [concs[3]]*9
# dictionary of regions that connects to each region's cells and the corresponding
# cables connected to that region
regions = {'LA':[la_cells, 'P1V', 'P01V'], 'LB':[lb_cells, 'P1V', 'P02V'],
           'LC':[lc_cells, 'P2V', 'P03V'], 'LD':[ld_cells, 'P2V', 'P04V'],
           'RA':[ra_cells, 'P3V', 'P05V'], 'RB':[rb_cells, 'P3V', 'P06V'],
           'RC':[rc_cells, 'P4V', 'P07V'], 'RD':[rd_cells, 'P4V', 'P08V']}

# construct the list of pin numbers in the pinout format
p0x_pins = np.arange(1, N_p0x_pins+1, 1)
p_odd_pins = ([i+1 for i in range(9)] + [86] + [i+1 for i in range(21, 29)] +
           [i+1 for i in range(43, 51)] + [i+1 for i in range(63, 71)] + [i+1 for i in range(86, 93)] +
           [43])
p_even_pins = ([i+1 for i in range(10, 19)] + [96] + [i+1 for i in range(31, 39)] +
               [i+1 for i in range(53, 61)] + [i+1 for i in range(73, 81)] +
               [i+1 for i in range(96, 103)] +[53])
p_odd_pins.extend([concs[3]]*(len(p0x_pins)-len(p_odd_pins)))
p_even_pins.extend([concs[3]]*(len(p0x_pins)-len(p_even_pins)))

cables = {'P1V' : {'P01V':[p0x_pins, p_odd_pins], 'P02V':[p0x_pins, p_even_pins]},
          'P2V' : {'P03V':[p0x_pins, p_odd_pins], 'P04V':[p0x_pins, p_even_pins]},
          'P3V' : {'P05V':[p0x_pins, p_odd_pins], 'P06V':[p0x_pins, p_even_pins]},
          'P4V' : {'P07V':[p0x_pins, p_odd_pins], 'P08V':[p0x_pins, p_even_pins]}}

# create array that orders cells based on pin diagram
l_array = np.zeros((N_rows, N_cols))
l_array[0][:6] = np.flip(cell_nums[:6]) # LA region
l_array[0][6:8] = cell_nums[6:8]
l_array[1][:8] = np.flip(cell_nums[8:16])
l_array[2][:8] = np.flip(cell_nums[16:24])
l_array[3][:8] = np.flip(cell_nums[24:32])

l_array[4][1:3] = np.flip(cell_nums[32:34]) # LB region
l_array[4][0] = np.flip(cell_nums[34])
l_array[4][3:8] = cell_nums[35:40]
l_array[5][:8] = cell_nums[40:48]
l_array[6][:8] = cell_nums[48:56]
l_array[7][:4] = np.flip(cell_nums[56:60])
l_array[7][4:8] = cell_nums[60:64]
l_array[8][:6] = np.flip(cell_nums[64:70])
l_array[8][6:8] = cell_nums[70:72]

l_array[9][:8] = np.flip(cell_nums[72:80]) # LC region
l_array[10][:8] = np.flip(cell_nums[80:88])
l_array[11][:8] = np.flip(cell_nums[88:96])
l_array[12][:8] = np.flip(cell_nums[96:104])
l_array[13][:8] = np.flip(cell_nums[104:112])

l_array[14][:8] = cell_nums[112:120] # LD region
l_array[15][:8] = cell_nums[120:128]
l_array[16][:8] = cell_nums[128:136]
l_array[17][6:8] = np.flip(cell_nums[136:138])
l_array[17][:6] = cell_nums[138:144]

flip_l_array = np.fliplr(l_array) # generate R region
r_array = np.where(flip_l_array>0, flip_l_array+144, flip_l_array)

cell_pin_array = l_array + r_array

# create array that orders cells using standard format
cell_order_array = np.zeros((N_rows, N_cols))
# print(cell_order_array)
for i in range(N_cells):
    y, x = grid_coords[i][0], grid_coords[i][1]
    cell_order_array[y][x] = cell_nums[i]

##################### Cell constructor functions ################################

def construct_cells(ifs, N_cells=N_cells, grid_coords=grid_coords,
                    cell_pin_array=cell_pin_array, regions=regions,
                    rc_labels=rc_labels):
    cells = []
    for i in range(N_cells):
        y, x = grid_coords[i][0], grid_coords[i][1]
        matching_pin_cell = int(cell_pin_array[y][x])
        for region, region_ls in regions.items():
            region_cells = region_ls[0]
    #         print('=======================')
    #         print('matching pin cell:', matching_pin_cell)
    #         print('Region:', region)
            if matching_pin_cell in region_cells:
                idx = region_cells.index(matching_pin_cell)
    #             print('Index', idx)
                p_cable_key, p0x_cable_key = region_ls[1], region_ls[2]
    #             print('keys', p_cable_key, p0x_cable_key)
                p_pin = cables[p_cable_key][p0x_cable_key][0][idx]
                p0x_pin = cables[p_cable_key][p0x_cable_key][1][idx]
    #             print('p_pin:', p_pin, 'p0x_pin:', p0x_pin)
                # print('cell:', i+1, 'label:', rc_labels[i])
                cell_ls = [i, i+1, ifs[i], alsis.rms(ifs[i]), alsis.ptov(ifs[i]),
                            grid_coords[i], rc_labels[i], region, matching_pin_cell,
                            p_cable_key, p_pin, p0x_cable_key, p0x_pin]
                cells.append(Cell(cell_ls))
            else: continue
    return cells


class Cell:

    def __init__(self, cell_ls):
        self.idx = cell_ls[0]
        self.no = cell_ls[1]
        self.ifunc = cell_ls[2]
        self.rms = cell_ls[3]
        self.pv = cell_ls [4]
        self.grid_coord = cell_ls[5]
        self.rc_label = cell_ls[6]
        self.region = cell_ls[7]
        self.pin_cell_no = cell_ls[8]
        self.p_cable = cell_ls[9]
        self.p_pin = cell_ls[10]
        self.p0x_cable = cell_ls[11]
        self.p0x_pin = cell_ls[12]

##################### Cell utility functions #################################

def cells_to_array(cells):
    return np.stack([cell.ifunc for cell in cells], axis=0)

def display_title_cell_no(ax, title, cell_no, fntsz):
    txtstring = title + '\nCell #: {}'.format(cell_no)
    title_plt_txt = ax.text(0.5, 1.075, txtstring, fontsize=fntsz,
                            ha='center', va='center', transform=ax.transAxes)
    return title_plt_txt

def displayStats(ax, cell, fntsz):
    stats_txtstring = "RMS: {:.2f} um\nPV: {:.2f} um".format(cell.rms, cell.pv)
    stats_plt_txt = ax.text(0.03, 0.05, stats_txtstring, fontsize=fntsz,
                            transform=ax.transAxes)
    return stats_plt_txt

def displayDetails(ax, cell, fntsz):
    details_txtstring = "Coordinates: {}\nRegion: {}\nPin cell #: {}\nCable: {},  Pin: {}\nCable: {},  Pin: {}".format(cell.rc_label, cell.region, cell.pin_cell_no, cell.p_cable, cell.p_pin, cell.p0x_cable, cell.p0x_pin)
    details_plt_txt = ax.text(0.68, 0.15, details_txtstring, fontsize=fntsz,
                                linespacing=2, ha='left', va='center',
                                transform=ax.transAxes)
    return details_plt_txt

def displayRadii(ax, ax_fntsz):
    large_R_text = ax.text(0.5, 0.075, 'Larger R', fontsize=ax_fntsz, color='red',
                            ha='center', va='center', transform=ax.transAxes)
    small_R_text = ax.text(0.5, 0.9255, 'Smaller R', fontsize=ax_fntsz, color='red',
                            ha='center', va='center', transform=ax.transAxes)

def displayIFs(cells, dx, imbounds=None, vbounds=None, colormap='jet',
            figsize=(8,6), title_fntsz=14, ax_fntsz=12,
            title='Influence Functions',
            x_title='Azimuthal Dimension (mm)',
            y_title='Axial Dimension (mm)',
            cbar_title='Figure (microns)',
            frame_time=500, repeat_bool=False, dispR=False,
            stats=False, details=False):
    """
    Displays set of IFs in a single animation on one figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(x_title, fontsize = ax_fntsz)
    ax.set_ylabel(y_title, fontsize = ax_fntsz)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    extent = fp.mk_extent(cells[0].ifunc, dx)
    if not vbounds:
        ifs = np.stack(tuple(cell.ifunc for cell in cells), axis=0)
        vbounds = [np.nanmin(ifs), np.nanmax(ifs)]
    if imbounds:
        lbnd = imbounds[0] - 1
        ubnd = imbounds[1]
    else:
        lbnd = 0
        ubnd = len(cells)
    ims = []
    for i in range(lbnd, ubnd):
        # print(i)
        im = ax.imshow(cells[i+lbnd].ifunc, extent=extent, aspect='auto', cmap=colormap,
                        vmin=vbounds[0], vmax=vbounds[1])
        cell_no = cells[i+lbnd].no #i + 1 + lbnd
        title_plt_txt = display_title_cell_no(ax, title, cell_no, title_fntsz)
        stats_plt_txt = ax.text(0, 0, '')
        details_plt_txt = ax.text(0, 0, '')
        if stats:
            # stats_plt_text = displayStats(ax, cells[i+lbnd], ax_fntsz)
            stats_txtstring = "RMS: {:.2f} um\nPV: {:.2f} um".format(cells[i+lbnd].rms, cells[i+lbnd].pv)
            stats_plt_txt = ax.text(0.03, 0.05, stats_txtstring, fontsize=ax_fntsz,
                                    transform=ax.transAxes)
        if details:
            details_plt_txt = displayDetails(ax, cells[i+lbnd], ax_fntsz)
        ims.append([im, title_plt_txt, stats_plt_txt, details_plt_txt])
        # print('appended:', i)
    cbar = fig.colorbar(ims[0][0], cax=cax)
    cbar.set_label(cbar_title, fontsize=ax_fntsz)
    if dispR:
        displayRadii(ax, ax_fntsz)
    ani = animation.ArtistAnimation(fig, ims, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    fps = int(1 / (frame_time/1000))

    return ani, fps

# class Cell:
#
#     @classmethod
#     def construct_cls_vars(cls):
#         cls.N_cells = 288 # number of cells
#         cls.N_rows = 18 # number of cls.rows (axial dimension)
#         cls.N_cols = 16 # number of columns (azimuthal dimension)
#         cls.N_ppins = 104 # number of pins on a P cable
#         cls.N_p0x_pins = 51 # number of pins on a P0X cable
#         cls.N_region_pins = 40 # number of pins per region
#
#         cls.cell_nums = np.arange(1, cls.N_cells+1, 1)
#         cls.rows = np.arange(0, cls.N_rows, 1)
#         cls.cols = np.arange(0, cls.N_cols, 1)
#         print(cls.rows[0])
#         cls.grid_coords = [[cls.rows[i%cls.N_rows],cls.cols[i//cls.N_rows]] for i in range(cls.N_cells)]
#         cls.rc_labels = ['R'+str(coord[0]+1)+'C'+str(coord[1]+1) for coord in cls.grid_coords]
#
#         cls.concs = ['GND', 'AOUT-XX', 'AGND', 'NC']
#         cls.la_cells = [cls.concs[0], cls.concs[1]] + cls.cell_nums[0:32].tolist() + [cls.concs[1]]*7 + [cls.concs[2]] + [cls.concs[3]]*9
#         cls.lb_cells = [cls.concs[2]] + cls.cell_nums[32:72].tolist() + [cls.concs[2]] + [cls.concs[3]]*9
#         cls.lc_cells = [cls.concs[2]] + cls.cell_nums[72:111].tolist() + [cls.concs[1], cls.concs[2]] + [cls.concs[3]]*9
#         cls.ld_cells = [cls.concs[2]] + cls.cell_nums[111:144].tolist() + [cls.concs[1]]*7 + [cls.concs[2]] + [cls.concs[3]]*9
#         cls.ra_cells = [cls.concs[2]] + [cls.concs[1]]*7 + np.flip(cls.cell_nums[144:176]).tolist() + [cls.concs[1], cls.concs[0]] + [cls.concs[3]]*9
#         cls.rb_cells = [int(cls.cell_nums[215])] + [cls.concs[1]] + np.flip(cls.cell_nums[176:215]).tolist() + [cls.concs[2]] + [cls.concs[3]]*9
#         cls.rc_cells = [cls.concs[2], cls.concs[1]] + np.flip(cls.cell_nums[216:255]).tolist() + [cls.concs[2]] + [cls.concs[3]]*9
#         cls.rd_cells = [cls.concs[2]] + [cls.concs[1]]*7 + np.flip(cls.cell_nums[255:288]).tolist() + [cls.concs[2]] + [cls.concs[3]]*9
#
#         cls.regions = {'LA':[cls.la_cells, 'P1V', 'P01V'], 'LB':[cls.lb_cells, 'P1V', 'P02V'],
#                    'LC':[cls.lc_cells, 'P2V', 'P03V'], 'LD':[cls.ld_cells, 'P2V', 'P04V'],
#                    'RA':[cls.ra_cells, 'P3V', 'P05V'], 'RB':[cls.rb_cells, 'P3V', 'P06V'],
#                    'RC':[cls.rc_cells, 'P4V', 'P07V'], 'RD':[cls.rd_cells, 'P4V', 'P08V']}
#
#         cls.p0x_pins = np.arange(1, cls.N_p0x_pins+1, 1)
#         cls.p_odd_pins = ([i+1 for i in range(9)] + [86] + [i+1 for i in range(21, 29)] +
#                    [i+1 for i in range(43, 51)] + [i+1 for i in range(63, 71)] + [i+1 for i in range(86, 93)] +
#                    [43])
#         cls.p_even_pins = ([i+1 for i in range(10, 19)] + [96] + [i+1 for i in range(31, 39)] +
#                        [i+1 for i in range(53, 61)] + [i+1 for i in range(73, 81)] +
#                        [i+1 for i in range(96, 103)] +[53])
#         cls.p_odd_pins.extend([cls.concs[3]]*(len(cls.p0x_pins)-len(cls.p_odd_pins)))
#         cls.p_even_pins.extend([cls.concs[3]]*(len(cls.p0x_pins)-len(cls.p_even_pins)))
#
#         cls.cables = {'P1V' : {'P01V':[cls.p0x_pins, cls.p_odd_pins], 'P02V':[cls.p0x_pins, cls.p_even_pins]},
#                   'P2V' : {'P03V':[cls.p0x_pins, cls.p_odd_pins], 'P04V':[cls.p0x_pins, cls.p_even_pins]},
#                   'P3V' : {'P05V':[cls.p0x_pins, cls.p_odd_pins], 'P06V':[cls.p0x_pins, cls.p_even_pins]},
#                   'P4V' : {'P07V':[cls.p0x_pins, cls.p_odd_pins], 'P08V':[cls.p0x_pins, cls.p_even_pins]}}
#
#         # create array that orders cells based on pin diagram
#         cls.l_array = np.zeros((cls.N_rows, cls.N_cols))
#         cls.l_array[0][:6] = np.flip(cls.cell_nums[:6]) # LA region
#         cls.l_array[0][6:8] = cls.cell_nums[6:8]
#         cls.l_array[1][:8] = np.flip(cls.cell_nums[8:16])
#         cls.l_array[2][:8] = np.flip(cls.cell_nums[16:24])
#         cls.l_array[3][:8] = np.flip(cls.cell_nums[24:32])
#
#         cls.l_array[4][1:3] = np.flip(cls.cell_nums[32:34]) # LB region
#         cls.l_array[4][0] = np.flip(cls.cell_nums[34])
#         cls.l_array[4][3:8] = cls.cell_nums[35:40]
#         cls.l_array[5][:8] = cls.cell_nums[40:48]
#         cls.l_array[6][:8] = cls.cell_nums[48:56]
#         cls.l_array[7][:4] = np.flip(cls.cell_nums[56:60])
#         cls.l_array[7][4:8] = cls.cell_nums[60:64]
#         cls.l_array[8][:6] = np.flip(cls.cell_nums[64:70])
#         cls.l_array[8][6:8] = cls.cell_nums[70:72]
#
#         cls.l_array[9][:8] = np.flip(cls.cell_nums[72:80]) # LC region
#         cls.l_array[10][:8] = np.flip(cls.cell_nums[80:88])
#         cls.l_array[11][:8] = np.flip(cls.cell_nums[88:96])
#         cls.l_array[12][:8] = np.flip(cls.cell_nums[96:104])
#         cls.l_array[13][:8] = np.flip(cls.cell_nums[104:112])
#
#         cls.l_array[14][:8] = cls.cell_nums[112:120] # LD region
#         cls.l_array[15][:8] = cls.cell_nums[120:128]
#         cls.l_array[16][:8] = cls.cell_nums[128:136]
#         cls.l_array[17][6:8] = np.flip(cls.cell_nums[136:138])
#         cls.l_array[17][:6] = cls.cell_nums[138:144]
#
#         cls.flip_l_array = np.fliplr(cls.l_array) # generate R region
#         cls.r_array = np.where(cls.flip_l_array>0, cls.flip_l_array+144, cls.flip_l_array)
#
#         cls.cell_pin_array = cls.l_array + cls.r_array
#
#     def __init__(self, ifunc):
#         for i in range(Cell.N_cells):
#             self.y, self.x = Cell.grid_coords[i][0], Cell.grid_coords[i][1]
#             # print(y,x)
#             self.matching_pin_cell = int(Cell.cell_pin_array[self.y][self.x])
#
#             for region, region_ls in Cell.regions.items():
#                 self.region_cells = region_ls[0]
#                 print('=======================')
#                 print('coords:', [self.y,self.x])
#                 print('matching pin cell:', self.matching_pin_cell)
#                 print('Region:', region)
#                 if self.matching_pin_cell in self.region_cells:
#                     self.idx = self.region_cells.index(self.matching_pin_cell)
#         #             print('Index', idx)
#                     self.p_cable_key, self.p0x_cable_key = region_ls[1], region_ls[2]
#         #             print('keys', p_cable_key, p0x_cable_key)
#                     self.p_pin = Cell.cables[self.p_cable_key][self.p0x_cable_key][0][self.idx]
#                     self.p0x_pin = Cell.cables[self.p_cable_key][self.p0x_cable_key][1][self.idx]
#         #             print('p_pin:', p_pin, 'p0x_pin:', p0x_pin)
#                     self.cell_idx = i
#                     self.cell_no = i+1
#                     self.grid_coord = Cell.grid_coords[i]
#                     self.rc_label = Cell.rc_labels[i]
#                     self.region = region
#                     self.pin_cell_no = self.matching_pin_cell
#                     self.p_cable = self.p_cable_key
#                     self.p_pin = self.p_pin
#                     self.p0x_cable = self.p0x_cable_key
#                     self.p0x_pin = self.p0x_pin
#                     self.ifunc = ifunc
#                 else: continue
