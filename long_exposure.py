import cv2
import numpy as np


def video2pic(input_file, output_file, mode):
    r_calc, g_calc, b_calc = None, None, None
    total = 0
    vid = cv2.VideoCapture(input_file)

    while True:
        ret, frame = vid.read()

        if ret is False:
            break

        total += 1

        b, g, r = cv2.split(frame.astype("float"))

        if r_calc is None:
            r_calc = r
            g_calc = g
            b_calc = b
        else:
            if mode == "stream":
                r_calc = cv2.accumulateWeighted(r, r_calc, 1 / total)
                g_calc = cv2.accumulateWeighted(g, g_calc, 1 / total)
                b_calc = cv2.accumulateWeighted(b, b_calc, 1 / total)
            elif mode == "firefly":
                r_calc = np.maximum(r_calc, r)
                g_calc = np.maximum(g_calc, g)
                b_calc = np.maximum(b_calc, b)
            else:
                raise ValueError("illegal mode.")

    if total == 0:
        raise ValueError("might be vid.read() error.")

    avg = cv2.merge([b_calc, g_calc, r_calc]).astype("uint8")
    cv2.imwrite(output_file, avg)


# video2pic(r".\input\firefly.mp4",
#           r".\output\firefly.png",
#           "firefly")
#
# video2pic(r".\input\stream.mp4",
#           r".\output\stream.png",
#           "stream")
