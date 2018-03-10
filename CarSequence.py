import xml.etree.ElementTree as ElementTree

import collections
import cv2

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


def convert_to_rect(rect_string):
    return Rectangle(int(rect_string[0]), int(rect_string[1]),
                     int(rect_string[2]) - int(rect_string[0]),
                     int(rect_string[3]) - int(rect_string[1]))


class CarSequence:
    def __init__(self, video_path, frame_numbers, rectangles, car_id):
        self._frame_numbers = frame_numbers
        self._frame_rectangles = rectangles
        self._id = car_id
        self._video_path = video_path

        self._frames = []
        self._rectangles = []
        self._iter = -1

    def get_frame(self):
        if len(self._frames) == 0:
            raise print('call read_frames first!')
        if self._iter == -1:
            print('Call next')

        return self._frames[self._iter]

    def get_rectangle(self):
        if len(self._frames) == 0:
            raise print('call read_frames first!')
        if self._iter == -1:
            print('Call next')
        return self._rectangles[self._iter]

    def get_id(self):
        return self._id

    def read_frames(self):
        # noinspection PyArgumentList
        cap = cv2.VideoCapture(self._video_path)
        for i in self._frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            self._frames.append(frame)
            self._rectangles.append(convert_to_rect(self._frame_rectangles[str(i)]))
        cap.release()

    def next(self):
        if len(self._frames) == 0:
            print('call read_frames first!')
            return False

        if self._iter + 1 < len(self._frames):
            self._iter += 1
            return True
        else:
            return False

    def reset(self):
        self._iter = -1

    def get_frame_num(self):
        return self._iter

    @staticmethod
    def parse_xml(xml_file, video_file):

        xml = ElementTree.parse(xml_file)

        root = xml.getroot()
        video_length = int(root.attrib['lastframe'])

        # Search cars in xml
        cars = []
        for carObj in root.findall("track[@label='Car']"):
            # Get attributes from carObject
            car_obj_name = carObj.attrib['id']
            start_frame = -1
            rectangles = {}

            for attribs in carObj.findall('box'):

                # Add the box if exist
                if attribs.attrib['xtl'] is not None:
                    frame = attribs.attrib['frame']
                    rectangle = [attribs.attrib['xtl'], attribs.attrib['ytl'], attribs.attrib['xbr'],
                                 attribs.attrib['ybr']]
                    rectangles[str(frame)] = rectangle

                # Count the frames which the object is
                if int(attribs.attrib['outside']) == 0 and int(attribs.attrib['occluded']) == 0:
                    if start_frame == -1:
                        start_frame = int(attribs.attrib['frame'])
                else:
                    if start_frame != -1:
                        end_frame = int(attribs.attrib['frame'])
                        if start_frame != end_frame:
                            cars.append(
                                CarSequence(video_file, range(start_frame, end_frame), rectangles, car_obj_name))

                        start_frame = -1

            if start_frame != -1:
                cars.append(CarSequence(video_file, range(start_frame, video_length), rectangles, car_obj_name))

        return cars


def clock(currentTime, timeOffset):
    newTime = currentTime
    tempMin = currentTime[1]
    tempMin += timeOffset
    # Check minutes
    if timeOffset >= 0:
        if tempMin < 60:
            newTime[1] = tempMin
        else:
            newTime[0] += 1
            newTime[1] = tempMin - 60
            if currentTime[0] > 23:
                newTime[0] = 0
    else:
        if tempMin <= 0:
            newTime[1] = 60 - abs(tempMin)
            newTime[0] -= 1
            if currentTime[0] < 0:
                newTime[0] = 23
        else:
            newTime[1] = tempMin

    return newTime


def getFrameObjectsWithBB(carObjects, frameNum, img):
    carCounter = []
    for carObj in carObjects:
        frameLists = carObj.get_frames()

        for frameList in frameLists:
            for frame in frameList:
                # if the car is on the actual frame
                if (frame == frameNum):
                    carCounter.append(carObj.get_id())
                    # check the saved rectangles
                    maxKey = 0
                    for key, value in carObj.get_rectangles().items():
                        if maxKey <= int(key) and int(key) <= frameNum:
                            maxKey = int(key)

                    value = carObj.get_rectangles()[str(maxKey)]
                    x = int(value[0])
                    y = int(value[1])
                    w = int(value[2]) - int(value[0])
                    h = int(value[3]) - int(value[1])

                    if (w != 0 and h != 0):
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, "Car ID #{}".format(carObj.get_id()), (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img, carCounter


def getFrameObjects(carObjects, frameNum):
    carCounter = []
    for carObj in carObjects:
        frameLists = carObj.get_frames()

        for frameList in frameLists:
            for frame in frameList:
                # if the car is on the actual frame
                if (frame == frameNum):
                    carCounter.append(carObj.get_id())

    return carCounter


def getAreaObjects(carObjects, frameNum, Area):
    carCounter = []
    for carObj in carObjects:
        frameLists = carObj.get_frames()

        for frameList in frameLists:
            for frame in frameList:
                # if the car is on the actual frame
                if (frame == frameNum):
                    # check the saved rectangles
                    maxKey = 0
                    for key, value in carObj.get_rectangles().items():
                        if maxKey <= int(key) and int(key) <= frameNum:
                            maxKey = int(key)

                    value = carObj.get_rectangles()[str(maxKey)]
                    x = int(value[0])
                    y = int(value[1])

                    if (x > Area[0] and y > Area[1] and x < Area[0] + Area[2] and y < Area[1] + Area[3]):
                        carCounter.append(carObj.get_id())

    return carCounter


def ROISelector(videoName, win_name):
    cap = cv2.VideoCapture(videoName)
    ret, image = cap.read()
    clone = image.copy()
    d = {'rect_pts': []}  # Starting and ending points

    def select_points(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            d['rect_pts'] = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            d['rect_pts'].append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(clone, d['rect_pts'][0], d['rect_pts'][1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        # display the image and wait for a keypress
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"):  # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord("c"):  # Hit 'c' to confirm the selection
            break

    # close the open windows
    cap.release()
    cv2.destroyWindow(win_name)
    rect = [d['rect_pts'][0][0], d['rect_pts'][0][1], d['rect_pts'][1][0] - d['rect_pts'][0][0],
            d['rect_pts'][1][1] - d['rect_pts'][0][1]]

    return rect


def processTrainData(carSet, timeSet):
    # This function creates a dictionary from two given sets.
    # The key is the time e.g. 12:00 in str, and its value is the number of cars on the same location timelapses e.g. [0,10,4,5] in a list.

    timeDict = dict()

    for timeListidx in range(len(timeSet)):
        timeList = timeSet[timeListidx]
        for ind in range(len(timeList)):

            actualTime = timeList[ind]
            actualCarNum = carSet[timeListidx][ind]

            if actualTime in timeDict:
                timeDict[actualTime].append(actualCarNum)
            else:
                timeDict[actualTime] = [actualCarNum]

    return timeDict


def measureAccuracy(groundTruth, prediction, tolerance):
    # This function measures the accuracy of the trained model within a tolerance range.
    Acc = 0

    for Idx in range(len(groundTruth)):
        tPercentage = groundTruth[Idx] * tolerance
        if groundTruth[Idx] + tPercentage >= prediction[Idx] >= groundTruth[Idx] - tPercentage:
            Acc += 1
    print(Acc / len(groundTruth))
    print(Acc // len(groundTruth))

    Acc = Acc / len(groundTruth)

    print('The trained model accuracy is: ' + str(Acc) + '%.')
