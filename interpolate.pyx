from __future__ import print_function
from scipy.ndimage import measurements, distance_transform_edt
from skimage import draw
from time import time
import json
import numpy as np
cimport cython
cimport numpy as np

DTYPE=np.uint64
ctypedef np.uint64_t DTYPE_t

# max average distance in pixels of vertices of one polygon to another to 
# consider them as part of the same process
POLYGON_MATCHING_THRESHOLD=100
POLYGON_MAX_STRETCH=20

@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b):
    '''Interpolate the labels between two label sections a and b.'''

    print("a.shape == ", a.shape[0], a.shape[1])
    print("b.shape == ", b.shape[0], b.shape[1])

    assert(a.shape[0] == b.shape[0])
    assert(a.shape[1] == b.shape[1])
    assert(a.dtype == DTYPE)
    assert(b.dtype == DTYPE)

    cdef int height = a.shape[0]
    cdef int width = a.shape[1]
    cdef int x, y, off_y, off_x, off_yd2, off_xd2

    cdef np.ndarray[DTYPE_t, ndim=2] interpolation = np.zeros([height, width], dtype=DTYPE)
    interpolation[:] = np.uint64(-1)

    # get bounding boxes of label components
    # • take care of repeating labels
    label_bbs_a = get_label_bbs(a)
    label_bbs_b = get_label_bbs(b)

    label_polygons_a = get_label_polygons(a, label_bbs_a)
    label_polygons_b = get_label_polygons(b, label_bbs_b)

    start = time()
    print("Interpolating polygons")

    interpolated = []

    # for each label component in a
    cdef DTYPE_t label
    for label in label_polygons_a.keys():

        polygons_a = label_polygons_a[label]
        bbs_a      = label_bbs_a[label]

        print("Processing label " + str(label))

        # if label doesn't exist in b, continue
        if label not in label_polygons_b:
            #print("Skipping (doesn't exist in both sections)")
            continue

        polygons_b = label_polygons_b[label]
        bbs_b      = label_bbs_b[label]

        # for each polygon in a, find all matching polygons in b
        for (polygon_a, bb_a) in zip(polygons_a, bbs_a):
            for (polygon_b, bb_b) in zip(polygons_b, bbs_b):

                (filtered_a, filtered_b) = truncate_polygons(polygon_a, polygon_b, bb_a, bb_b)

                # too far away from each other
                if len(filtered_a[0])*len(filtered_b[0]) == 0:
                    continue

                (score, mapping) = match_polygons(filtered_a, filtered_b)

                # if too far away, continue
                if score > POLYGON_MATCHING_THRESHOLD:
                    continue

                polygon_c = interpolate_polygon(filtered_a, filtered_b, mapping)

                interpolated.append((len(polygon_c[0]), label, polygon_c))

    interpolated.sort()
    for (size, label, polygon) in interpolated:

        print("Drawing interpolated polygon of size " + str(size))
        rows, cols = draw.polygon(polygon[0], polygon[1])
        interpolation[rows, cols] = label

    print("Finished in " + str(time()-start) + "s")

    return interpolation

@cython.boundscheck(False)
@cython.wraparound(False)
def get_label_bbs(np.ndarray[DTYPE_t, ndim=2] a):

    cache = read_label_bbs(a.data)
    if cache is not None:
        print("Reading label components from cache")
        return cache

    start = time()

    cdef int height = a.shape[0]
    cdef int width = a.shape[1]

    label_bbs = {}

    labels = np.unique(a)
    print("Getting label components for " + str(len(labels)) + " labels")

    for label in labels:

        # some of the special labels don't count
        if label >= np.uint64(-10):
            continue

        (compontents, num) = measurements.label((a==label))
        label_bbs[label] = measurements.find_objects(compontents)

    print("Finished in " + str(time() - start) + "s")

    save_label_bbs(label_bbs, a.data)

    return label_bbs

def save_label_bbs(data_dict, reference):

    d = { str(k): [ [ (s.start, s.stop) for s in c ] for c in v ] for (k,v) in data_dict.iteritems() }

    print("hash is " + str(hash(reference)))

    with open('cache_' + str(hash(reference)) + '_label_bbs.json', 'w') as f:
        json.dump(d, f)

def read_label_bbs(reference):

    try:

        with open('cache_' + str(hash(reference)) + '_label_bbs.json', 'r') as f:
            d = json.load(f)

        return { np.uint64(k) : [ [ slice(s[0], s[1]) for s in c ] for c in v ] for (k,v) in d.iteritems() }

    except Exception as e:

        print(e)
        return None

@cython.boundscheck(False)
@cython.wraparound(False)
def get_label_polygons(np.ndarray[DTYPE_t, ndim=2] a, label_bbs):
    '''NO BOUNDARY CHECKS.

    This implementation assumes that all components do not touch the boundary.
    '''

    start = time()
    print("Getting label polygones")

    # coordinates of left and right hand
    cdef int rh_x, rh_y, lh_x, lh_y, first_lh_x, first_lh_y, first_rh_x, first_rh_y
    cdef int start_x, start_y, stop_x, stop_y
    cdef int direction
    cdef int UP = 0
    cdef int RIGHT = 1
    cdef int DOWN = 2
    cdef int LEFT = 3
    cdef DTYPE_t label

    polygons = {}

    for (label, components) in label_bbs.iteritems():
        polygons[label] = []
        for component in components:

            # component[0] is y, 1 is x
            start_y, stop_y = component[0].start, component[0].stop
            start_x, stop_x = component[1].start, component[1].stop

            # find a boundary pixel
            for first_rh_y in range(start_y, stop_y):
                for first_rh_x in range(start_x, stop_x):
                    if a[first_rh_y,first_rh_x] == label:
                        break
                else:
                    continue
                break

            direction = UP
            rh_x, rh_y = first_rh_x, first_rh_y
            lh_x, lh_y = rh_x - 1, rh_y

            first_lh_x, first_lh_y = lh_x, lh_y

            polygon_x = []
            polygon_y = []

            # walk with right hand on the wall
            first_point = True
            while (lh_x, lh_y) != (first_lh_x, first_lh_y) or first_point:

                # add current point to polygon
                polygon_x.append(0.5*(lh_x+rh_x))
                polygon_y.append(0.5*(lh_y+rh_y))
                first_point = False

                # try to go in current direction
                if direction == UP:
                    lh_y -= 1
                    rh_y -= 1
                if direction == RIGHT:
                    lh_x += 1
                    rh_x += 1
                if direction == DOWN:
                    lh_y += 1
                    rh_y += 1
                if direction == LEFT:
                    lh_x -= 1
                    rh_x -= 1

                # still right hand on wall and left hand in air?
                if a[rh_y,rh_x] == label and a[lh_y,lh_x] != label:
                    continue

                # Nope. Then there are two cases: we ran into the wall, or we 
                # lost touch. If both happened, we found two touching corners, 
                # which are handled just like losing touch.

                # we lost touch
                if a[rh_y,rh_x] != label:
                    # our right hand becomes the new left hand
                    lh_x, lh_y = rh_x, rh_y
                    # we turn by 90 degree to the right
                    direction = (direction + 1)%4
                    if direction == UP:
                        rh_x += 1
                    if direction == RIGHT:
                        rh_y += 1
                    if direction == DOWN:
                        rh_x -= 1
                    if direction == LEFT:
                        rh_y -= 1

                # we ran into the wall:
                elif a[lh_y,lh_x] == label:
                    # our left hand becomes the new right hand
                    rh_x, rh_y = lh_x, lh_y
                    # we turn by 90 degree to the left
                    direction = (direction - 1)%4
                    if direction == UP:
                        lh_x -= 1
                    if direction == RIGHT:
                        lh_y -= 1
                    if direction == DOWN:
                        lh_x += 1
                    if direction == LEFT:
                        lh_y += 1

            polygons[label].append(np.array([polygon_y, polygon_x], dtype=np.float))

        assert(len(polygons[label]) == len(label_bbs[label]))

    print("Finished in " + str(time()-start) + "s")


    return polygons

@cython.boundscheck(False)
@cython.wraparound(False)
def truncate_polygons(np.ndarray[np.float_t, ndim=2] polygon_a, np.ndarray[np.float_t, ndim=2] polygon_b, bb_a, bb_b):
    '''Remove all points from the given polygons that are too far away from the other one.'''

    cdef np.ndarray[np.float_t, ndim=2] points_a = np.array(polygon_a).transpose()
    cdef np.ndarray[np.float_t, ndim=2] points_b = np.array(polygon_b).transpose()

    cdef np.ndarray[np.float_t, ndim=2] filtered_a = filter_points(points_a, points_b, bb_b)
    cdef np.ndarray[np.float_t, ndim=2] filtered_b = filter_points(points_b, points_a, bb_a)

    return (filtered_a.transpose(), filtered_a.transpose())

def filter_points(np.ndarray[np.float_t, ndim=2] points_a, np.ndarray[np.float_t, ndim=2] points_b, bb_b):

    # size of distance map
    size = tuple(bb_b[d].stop-bb_b[d].start+2*POLYGON_MAX_STRETCH for d in range(2))
    # upper left of distance map in world coordinates
    offset = tuple(bb_b[d].start-POLYGON_MAX_STRETCH for d in range(2))

    distances = np.ones(size, dtype=np.float)
    for p in points_b:
        distances[p[0]-offset[0],p[1]-offset[1]] = 0

    distances = distance_transform_edt(distances)

    len_a = len(points_a)
    keep = np.zeros((len_a,), dtype=np.bool)
    for i in range(len_a):
        y = points_a[i][0]-offset[0]
        x = points_a[i][1]-offset[1]
        if y < 0 or x < 0 or y >= size[0] or x >= size[1]:
            continue
        if distances[y,x] < POLYGON_MAX_STRETCH:
            keep[i] = True
    return points_a[keep]

@cython.boundscheck(False)
@cython.wraparound(False)
def match_polygons(np.ndarray[np.float_t, ndim=2] polygon_a, np.ndarray[np.float_t, ndim=2] polygon_b):
    '''Match two polygons, each given as [Y,X], where Y and X are lists of y and 
    x coordinates.

    Points on the polygon are assumed to be equidistantly spaced.

    Returns (score, matches), the matching score and a mapping of vertices of 
    polygon_a to polygon_b.
    '''

    # find the smaller one
    cdef int len_a = len(polygon_a[0])
    cdef int len_b = len(polygon_b[0])
    cdef float score

    if len_a > len_b:
        (score, mapping) = match_polygons(polygon_b, polygon_a)
        # mapping is now from b to a, we have to revert it
        return (score, [ (a,b) for (b,a) in mapping ])

    # now a is smaller than b

    cdef np.ndarray[np.float_t, ndim=2] points_a = np.array(polygon_a).transpose()
    cdef np.ndarray[np.float_t, ndim=2] points_b = np.array(polygon_b).transpose()

    # subsample b
    b_sub_indices = [ int(round(t*len_b)) for t in np.linspace(0.0, 1.0, len_a, endpoint=False) ]

    #print("\tpoints_a: " + str(points_a))
    #print("\tpoints_b: " + str(points_b))
    #print("\tb sub ind: " + str(b_sub_indices))

    # find best matching
    cdef float best_score = -1
    cdef int best_permutation = 0
    cdef int i, j
    cdef int len_sub_b = len(b_sub_indices)
    for permutation in range(len_a):
        score = 0
        #print("\tchecking permutation " + str(permutation))
        for i in range(len_a):
            j = b_sub_indices[(i+permutation)%len_sub_b]
            #print("\t\tpoint " + str(i) + " in a: " + str(points_a[i]))
            #print("\t\tpoint " + str(j) + " in b: " + str(points_b[j]))
            #print("\t\tdifference (a-b): " + str(points_a[i]-points_b[j]))
            score += np.linalg.norm(points_a[i]-points_b[j])
        #print("\tfor permutation " + str(permutation) + ", score is " + str(score))
        if best_score < 0 or score < best_score:
            best_score = score
            best_permutation = permutation

    best_score /= len_a
    best_mapping = [ (i,b_sub_indices[(i+best_permutation)%len_sub_b]) for i in range(len_a) ]

    return (best_score, best_mapping)

@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate_polygon(np.ndarray[np.float_t, ndim=2] polygon_a, np.ndarray[np.float_t, ndim=2] polygon_b, mapping):

    # find the smaller one
    cdef int len_a = len(polygon_a[0])
    cdef int len_b = len(polygon_b[0])

    if len_a > len_b:
        return interpolate_polygon(polygon_b, polygon_a, [ (b,a) for (a,b) in mapping ])

    # now we can assume that indices of a in mapping are contiguous

    return [
            [ va+(vb-va)*0.5 for (va,vb) in zip(polygon_a[d], [polygon_b[d][mapping[i][1]] for i in range(len_a) ]) ]
            for d in range(2)
    ]

def center(component):
    return np.array([ float(s.start - s.stop)/2 for s in component ])

def height(component):
    return component[0].stop - component[0].start

def width(component):
    return component[1].stop - component[1].start
