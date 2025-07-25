import cv2
import numpy as np


def video2pic(input_file, output_file, mode):
    rCalc, gCalc, bCalc = None, None, None
    total = 0
    vid = cv2.VideoCapture(input_file)

    while True:
        ret, frame = vid.read()

        if ret is False:
            break

        total += 1

        B, G, R = cv2.split(frame.astype("float"))

        if rCalc is None:
            rCalc = R
            gCalc = G
            bCalc = B
        else:
            if mode == "stream":
                rCalc = cv2.accumulateWeighted(R, rCalc, 1 / total)
                gCalc = cv2.accumulateWeighted(G, gCalc, 1 / total)
                bCalc = cv2.accumulateWeighted(B, bCalc, 1 / total)
            elif mode == "firefly":
                rCalc = np.maximum(rCalc, R)
                gCalc = np.maximum(gCalc, G)
                bCalc = np.maximum(bCalc, B)
            else:
                raise ValueError("Illegal Mode")

    if ret is False:
        raise ValueError("Reading Video Failed!")

    avg = cv2.merge([bCalc, gCalc, rCalc]).astype("uint8")
    cv2.imwrite(output_file, avg)


video2pic(r"D:\github\photo_long_exposure\input_file\firefly.mp4",
          r"D:\github\photo_long_exposure\output_file\firefly.png",
          "firefly")
#
# video2pic(r"D:\github\photo_long_exposure\input\stream.mp4",
#           r"D:\github\photo_long_exposure\output\stream.png",
#           "stream")
