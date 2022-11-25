import os
import numpy as np
import cv2


mtl_attr_type = { \
    'FLOAT_LIST': ['Ka', 'Kd', 'Ks', 'Ke', 'Tf'], \
    'FLOAT': ['d', 'Ns', 'Ni', 'Tr'], \
    'INT': ['illum'], \
    'STRING': ['map_Ka', 'map_Kd', 'map_Ks', 'map_Ke', 'map_d', 'map_bump']}


def locate_file(name, dir_='.'):
    name_ = name
    if len(name) > 2 and (name[0] == '/' or \
                          (name[1:3] == ':\\' and name[0].lower() >= 'c' and name[0].lower() <= 'z')):
        if os.path.isfile(name):
            return name
        else:
            name_ = os.path.basename(name)
    name_ = os.path.join(dir_, name_)
    if os.path.isfile(name_):
        return name_
    else:
        name_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(name_))
        if os.path.isfile(name_):
            return name_
        else:
            return name


def save_mtl(file_name, mtls, imgs={}):
    with open(file_name, 'w') as f:
        for mtl in mtls.keys():
            f.write('newmtl %s\n' % mtl)
            for attr in mtls[mtl].keys():
                if attr in mtl_attr_type['FLOAT_LIST']:
                    f.write(attr)
                    for i in mtls[mtl][attr]:
                        f.write(' %f' % i)
                    f.write('\n')
                elif attr in mtl_attr_type['FLOAT']:
                    f.write('%s %f\n' % (attr, mtls[mtl][attr]))
                elif attr in mtl_attr_type['INT']:
                    f.write('%s %d\n' % (attr, mtls[mtl][attr]))
                elif attr in mtl_attr_type['STRING']:
                    f.write('%s %s\n' % (attr, mtls[mtl][attr]))
    dir_ = os.path.dirname(file_name)
    for name in imgs.keys():
        if not imgs[name] is None:
            name_ = locate_file(name, dir_)
            cv2.imwrite(name_, imgs[name])
    return True


def load_obj(file_name, convert_to_triangle=False):
    if file_name[-4:].lower() != '.obj':
        return None
    with open(file_name, 'r') as f:
        text = f.readlines()
    v = [];
    vt = [];
    vn = []
    fv = [];
    ft = [];
    fn = []
    mtls = {};
    imgs = {}
    smooth = [];
    s = 0
    group = {'': []};
    g_name = ''
    f_mtl = {'': []};
    mtl_name = ''
    for line in text:
        line = [seg for seg in line.strip().split(' ') if len(seg) > 0]
        if len(line) >= 3 and line[0] == 'v':
            v += [[float(f) for f in line[1:]]]
        elif len(line) >= 2 and line[0] == 'vt':
            vt += [[float(f) for f in line[1:]]]
        elif len(line) >= 3 and line[0] == 'vn':
            vn += [[float(f) for f in line[1:]]]
        elif len(line) >= 1 and line[0] == 'usemtl':
            if len(line) >= 2:
                mtl_name = line[1]
            else:
                mtl_name = ''
            if not mtl_name in f_mtl.keys():
                f_mtl[mtl_name] = []
        elif len(line) >= 1 and line[0] == 'g':
            if len(line) >= 2:
                g_name = line[1]
            else:
                g_name = ''
            if not g_name in group.keys():
                group[g_name] = []
        elif len(line) >= 1 and line[0] == 's':
            if len(line) == 1:
                s_ = 0
            elif line[1].lower() == 'off':
                s_ = 0
            elif line[1].lower() == 'on':
                s_ = 1
            else:
                s_ = int(line[1])
            if s_ >= 0 and s_ <= 5:
                s = s_
        elif len(line) >= 4 and line[0] == 'f':
            f = [[], [], []]
            for i in range(1, len(line)):
                l = line[i].split('/')
                for j in range(3):
                    if j < len(l) and len(l[j]) > 0:
                        f[j] += [int(l[j]) - 1]
            if convert_to_triangle:
                fv += [[f[0][0], f[0][i - 1], f[0][i]] for i in range(2, len(f[0]))]
                ft += [[f[1][0], f[1][i - 1], f[1][i]] for i in range(2, len(f[1]))]
                fn += [[f[2][0], f[2][i - 1], f[2][i]] for i in range(2, len(f[2]))]
                smooth += [s] * (len(f[0]) - 2)
                group[g_name] += range(len(fv) - len(f[0]) + 1, len(fv))
                f_mtl[mtl_name] += range(len(fv) - len(f[0]) + 1, len(fv))
            else:
                fv += [f[0]]
                ft += [f[1]]
                fn += [f[2]]
                smooth += [s]
                group[g_name] += [len(fv) - 1]
                f_mtl[mtl_name] += [len(fv) - 1]
    v = np.array(v, 'float32')
    vt = np.array(vt, 'float32')
    vn = np.array(vn, 'float32')
    if vn.shape[0] > 0:
        norm_ = np.sum(vn * vn, 1)
        eps = 1e-12
        unregulize = np.logical_or( \
            np.logical_and(norm_ > eps, norm_ < 1 - eps), norm_ > 1 + eps)
        vn[unregulize, :] /= np.tile(np.sqrt(norm_[unregulize]).reshape((-1, 1)), (1, 3))
    if v.shape[1] > 3:
        color = v[:, 3:]
        v = v[:, :3]
        if color.max() > 1:
            color /= 255
        v = (v, vt, vn, color)
    elif len(vn) != 0:
        v = (v, vt, vn)
    elif len(vt) != 0:
        v = (v, vt)
    else:
        v = (v,)

    v_num = 0;
    t_num = 0;
    n_num = 0;
    s = True
    for i in range(len(fv)):
        if i == 0:
            v_num = len(fv[0])
            if len(ft) > 0:
                t_num = len(ft[0])
            if len(fn) > 0:
                n_num = len(fn[0])
            s = (smooth[0] == 0)
        else:
            if v_num != len(fv[i]): v_num = -1
            if len(ft) > i and t_num != len(ft[i]): vt_num = -1
            if len(fn) > i and n_num != len(fn[i]): vn_num = -1
            if smooth[i] != 0: s = False
    if v_num >= 3: fv = np.array(fv, 'int32')
    if t_num >= 3:
        ft = np.array(ft, 'int32')
    elif t_num == 0:
        ft = np.array([], 'int32')
    if n_num >= 3:
        fn = np.array(fn, 'int32')
    elif n_num == 0:
        fn = np.array([], 'int32')
    if s:
        smooth = np.array([], 'uint8')
    else:
        smooth = np.array(smooth, 'uint8')
    if len(fn) != 0:
        fv = (fv, ft, fn)
    elif len(ft) != 0:
        fv = (fv, ft)
    else:
        fv = (fv,)
    for k in list(f_mtl.keys()):
        if len(f_mtl[k]) == 0:
            del f_mtl[k]
    return v, fv, (f_mtl, mtls, imgs), smooth, group


def save_obj_vertex(file_name, v):
    vertex = v.reshape((-1,3))
    with open(file_name, 'w') as f:
        for i in range(vertex.shape[0]):
            f.write('v {} {} {}\n'.format(vertex[i, 0], vertex[i, 1], vertex[i, 2]))

    return


def save_obj(file_name, v, face, mtls=(), smooth=[]):
    if file_name[-4:].lower() != '.obj':
        return False
    if isinstance(v, tuple):
        if len(v) >= 4:
            color = v[3]
        else:
            color = []
        if len(v) >= 3:
            vn = v[2]
        else:
            vn = []
        if len(v) >= 2:
            vt = v[1]
        else:
            vt = []
        if len(v) >= 1:
            v = v[0]
        else:
            v = []
    else:
        color = vn = vt = []
    if isinstance(face, tuple):
        if len(face) >= 3:
            fn = face[2]
        else:
            fn = []
        if len(face) >= 2:
            ft = face[1]
        else:
            ft = []
        if len(face) >= 1:
            fv = face[0]
        else:
            fv = []
    else:
        fn = ft = []
        fv = face
    with open(file_name, 'w') as f:
        if isinstance(mtls, tuple) and len(mtls) >= 2 and len(mtls[1]) > 0:
            if len(mtls) >= 3:
                res = save_mtl(file_name[:-4] + '.mtl', mtls[1], mtls[2])
            else:
                res = save_mtl(file_name[:-4] + '.mtl', mtls[1])
            if res:
                f.write('mtllib %s.mtl\n' % os.path.basename(file_name[:-4]))
            f_mtls = mtls[0]
        elif len(mtls) == 0:
            f_mtls = {'': range(len(fv))}
        for i in range(len(v)):
            f.write(('v' + ' %f' * len(v[i])) % tuple(v[i]))
            if len(color) > i:
                f.write((' %f' * len(color[i])) % tuple(color[i]))
            f.write('\n')
        for i in range(len(vt)):
            f.write(('vt' + ' %f' * len(vt[i]) + '\n') % tuple(vt[i]))
        for i in range(len(vn)):
            f.write(('vn' + ' %f' * len(vn[i]) + '\n') % tuple(vn[i]))
        if len(smooth) > 0:
            s = smooth[0]
            f.write('s %d\n' % s)
        else:
            s = 0
        for k in f_mtls.keys():
            if k != '': f.write('usemtl %s\n' % k)
            for i in f_mtls[k]:
                if len(fv) <= i: continue
                if len(smooth) > i and smooth[i] != s:
                    s = smooth[i]
                    f.write('s %d\n' % s)
                v_num = len(fv[i])
                has_tex = (len(ft) > i and len(ft[i]) == v_num and len(vt) > 0)
                has_norm = (len(fn) > i and len(fn[i]) == v_num and len(vn) > 0)
                f.write('f')
                for j in range(v_num):
                    if has_tex and has_norm:
                        f.write(' %d/%d/%d' % (fv[i][j] + 1, ft[i][j] + 1, fn[i][j] + 1))
                    elif has_tex:
                        f.write(' %d/%d' % (fv[i][j] + 1, ft[i][j] + 1))
                    elif has_norm:
                        f.write(' %d//%d' % (fv[i][j] + 1, fn[i][j] + 1))
                    else:
                        f.write(' %d' % (fv[i][j] + 1))
                f.write('\n')
