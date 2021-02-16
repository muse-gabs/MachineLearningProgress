#%%
import cv2 as cv 
import matplotlib.pyplot as plt

net = cv.dnn.readNetFromTensorflow("graph_opt.pb") # weights

inWidth = 368
inHeight = 368
thr = 0.2

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

img = cv.imread("C:\\Users\\gabby\\AI and ML notes\\OpenCVProgramming\\testpose.jpg")

#plt.imshow(img) # this plots in BGR which is why we get the weird color scheme


# plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) # this will convert it to RGB

def pose_estimation(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    # we obtain 127.5 by taking 255 and dividing by 2
    # blobFromImage
    #     - input image
    #     - scalefactor (scales values by scalefactor)
    #     - spatial size for output image
    #     - scalar of mean values that are subtracted from channels (mean-B, mean-G, mean-R) 
    #       * BGR order because swapRB is true
    #     - set crop to false because it indicates whether image is cropped after resize 
    #       * don't want it to be cropped
    # this is all found from https://docs.opencv.org/master/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :] # MobileNet output [1, 57, -1, -1] (only first 19 elements)

    assert(len(BODY_PARTS) == out.shape[1])

    points = []

    for i in range(len(BODY_PARTS)):
        # obtain the corresponding body part
        heatMap = out[0, i, :, :]

        # obtain local maximums
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1])/ out.shape[2]
        # add the point if the confidence is higher than threshold
        points.append((int(x), int(y)) if conf > thr else None)

    # we had defined POSE_PAIRS above so we select where it begins
    # and where it ends to draw the lines/ellipses of the person's 
    # pose parts
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        # make sure the part exists in BODY_PARTS
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        # obtain the number that is associated to the
        # part
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        # specify the line and ellipse types to 
        # draw from the points that have been obtained
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3,3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3,3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return frame

estimated_image = pose_estimation(img)
plt.imshow(cv.cvtColor(estimated_image, cv.COLOR_BGR2RGB))


# %%
