
import sys
import os
import dlib
import glob
import numpy as np

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets) == 1:
        shape = predictor(img, dets[0])
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        points = []
        for i in lm_idx:
            point = shape.part(i)
            points.append([point.x, point.y])
        points = np.array(points)
        ldmk = np.stack([points[0],
                         np.mean(points[[1, 2], :], 0),
                         np.mean(points[[3, 4], :], 0),
                         points[5, :],
                         points[6, :]],
                        axis=0)
        ldmk = ldmk[[1, 2, 0, 3, 4], :].astype(np.str_)
        txt_path = os.path.join(os.path.dirname(f),'detections',os.path.basename(f)[:-4]+'.txt')
        with open(txt_path, 'w') as writer:
            for point in ldmk:
                writer.write(' '.join(point)+'\n')
                
