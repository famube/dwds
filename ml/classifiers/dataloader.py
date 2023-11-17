import sys
from scipy.sparse import csr_matrix
import pandas

def load_class_probabilities(filename):
    df = pandas.read_csv(filename, sep=" ")
    return list(df.values[:,1:])

def load_class_probabilities_df(filename):
    df = pandas.read_csv(filename, sep=" ")
    df = df.drop(labels="labels", axis="columns")
    return df

def load_class_f1(filename):
    res = {}
    infile = open(filename)
    line = infile.readline()
    while "precision" not in line:
        line = infile.readline()
    line = infile.readline()
    lin = infile.readline().strip()
    while lin != "":
        spl = lin.split()
        c = int(spl[0])
        f1 = float(spl[len(spl)-2])
        res[c] = f1
        lin = infile.readline().strip()
    infile.close()
    return res

def find_shape(filename):
    infile = open(filename)
    n = 0
    nlines = 0
    for line in infile:
        nlines += 1
        spl = line.strip().split()
        for s in spl[1:]:
            tok = s.split(":")
            featureid = int(tok[0]) - 1
            if featureid > n:
                n = featureid
    infile.close()
    return nlines, n + 1

def load_vocabulary(filename):
    infile = open(filename)
    dic = {}
    for line in infile:
        spl = line.strip().split()
        word = spl[0]
        idx = int(spl[1]) - 1
        dic[idx] = word
    infile.close()
    return dic


def load_sparse_data_dic(filename):
    infile = open(filename)
    y = []
    rows = []
    for line in infile:
        row = {}
        spl = line.strip().split()
        label = int(spl[0])
        y.append(label)
        for s in spl[1:]:
            tok = s.split(":")
            featureid = int(tok[0]) - 1
            value = float(tok[1])
            row[featureid] = value
        rows.append(row)
    infile.close()
    return rows,y


def load_sparse_data(filename, ndim=None):
    infile = open(filename)
    y = []
    rows = []
    cols = []
    values = []

    lineid = 0
    for line in infile:
        spl = line.strip().split()
        label = int(spl[0])
        y.append(label)
        for s in spl[1:]:
            tok = s.split(":")
            featureid = int(tok[0]) - 1
            value = float(tok[1])
            if featureid < 0:
                continue
            rows.append(lineid)
            cols.append(featureid)
            values.append(value)

        if ndim != None:
            rows.append(lineid)
            cols.append(ndim)
            values.append(0)

        lineid += 1
    infile.close()
    X = csr_matrix( (values, (rows, cols)) )
    return X,y



def load_sequence_data(filename):
    infile = open(filename)
    y = []
    X = []
    for line in infile:
        spl = line.strip().split()
        label = int(spl[0])
        y.append(label)
        X.append(spl[1:])
    infile.close()
    return X,y



def load_list(filename):
    infile = open(filename)
    res = [x.strip().split()[0] for x in infile]
    infile.close()
    return res

def load_classes(filename):
    infile = open(filename)
    res = [int(x.strip().split()[0]) for x in infile]
    infile.close()
    return res

