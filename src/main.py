from math import floor
from random import randint
import cv2
import time
import os
import sys
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from collections import Counter
import json
from itertools import permutations


global cmd

cmd = open("../cmd", "r").read().split('\n')


def readDataSet():
    data = json.load(open("../json/dataset.json", "r"))
    for k, v in data.items():
        # using eval not recommended
        data[k] = eval(v)
    return data


def readWordList():
    return json.load(open("../json/words.json", "r"))


class Objects:
    def __init__(self, name="screen.png", dataset=None):
        self.r_path = "../images/"
        # image path
        self.path = self.r_path+name
        if not os.path.exists(self.path):
            raise Exception("Path/File doesn't exist")
        # image
        self.img = None
        # image thresh
        self.thresh = None
        # plain image
        self.plane = None
        # image contours
        self.contours = []
        # option and answer boxes contour approx
        self.opt_boxes = []
        self.ans_boxes = []
        # option and answer boxes midpoints(centroids)
        self.opt_midpts = []
        self.ans_midpts = []
        # contour of letters in options(required) and answers(optional)
        self.opt_letters = []
        self.ans_letters = []
        # dataset
        self.dataset = dataset

    def readImage(self):
        self.img = cv2.imread(self.path)

    def drawPlane(self):
        self.plane = np.zeros((self.img.shape[:2]))

    def thresholdImage(self):
        # thresholding the image - to gray
        imgray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        # invert image
        self.thresh = 255-thresh
        # kernel = np.ones((3, 3), dtype=np.uint8)
        # self.thresh = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, kernel)
        # self.thresh = cv2.erode(self.thresh, kernel, iterations=1)

    def findContours(self):
        # find contours(borders/outlines)
        self.image, self.contours, self.hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def drawContours(self):
        cv2.drawContours(self.plane, self.contours, -1, (120, 105, 255), 3)

    def separateContours(self):
        # getting required contours
        def centroid(contour):
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return (cx, cy)

        opt_letters_hier = []
        flag = 0
        temp = []
        temp_hier = []
        self.perims = []

        def push(temp, temp_hier):
            self.opt_letters.append(temp[1:])
            opt_letters_hier.append(temp_hier[1])
            temp = []
            temp_hier = []
            return temp, temp_hier
        # [+-]2-error
        fixed_perim_1 = [617, 618, 619, 620]
        fixed_perim_2 = [681, 682, 683]
        derived_perim_3 = []
        # To auto select opt and ans boxes based on perimeters
        temp_flags = []
        pos = 0
        # START - algorithm to find boxes
        for idx, cnt in enumerate(self.contours):
            # find perimeter of contour(closed <- True)
            perim = floor(cv2.arcLength(cnt, True))
            self.perims.append(perim)
            if perim in fixed_perim_1 or perim in fixed_perim_2:
                temp_flags.append(1)
                pos = idx
            else:
                temp_flags.append(0)
            # for finish levels
        mem = dict()
        for idx, v in enumerate(temp_flags[pos+1:]):
            i = self.perims[idx+pos+1]
            # exists more than once
            if i in mem:
                mem[i] = mem[i-1] = mem[i+1] = (mem[i]+1)
                if mem[i] > 1:
                    derived_perim_3.extend([i-1, i, i+1])
                    break
            else:
                # 2-error
                mem[i] = mem[i-1] = mem[i+1] = 0
        # END - algorithm
        # some fix
        if 428 in derived_perim_3:
            derived_perim_3.append(588)

        for idx, cnt in enumerate(self.contours):
            perim = self.perims[idx]
            # answer
            if perim in derived_perim_3:
                self.ans_boxes.append(cnt)
                self.ans_midpts.append(centroid(cnt))
                if flag != 2:
                    temp, temp_hier = push(temp, temp_hier)
                flag = 2
            # option
            elif perim in fixed_perim_1:
                self.opt_boxes.append(cnt)
                self.opt_midpts.append(centroid(cnt))
                flag = 1
                if len(temp) != 0:
                    temp, temp_hier = push(temp, temp_hier)
            elif perim in fixed_perim_2:
                flag = 0
            elif flag == 1 and perim > 150:
                # contour approximation(Douglas-Peucker algorithm)
                epsilon = 0.01*cv2.arcLength(cnt, True)
                approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                temp.append(approx_cnt)
                temp_hier.append(self.hierarchy[0][idx])
        for cnt in self.opt_letters:
            cv2.drawContours(self.plane, cnt, -1, (120, 105, 255), 3)

    def writeIntoPlane(self):
        cv2.imwrite(self.r_path + "bin.png", self.plane)

    def getOptionLetterContours(self):
        return self.opt_letters

    def getOptionButton(self):
        return self.opt_midpts

    def getAnswerButton(self):
        return self.ans_midpts

    def randomRGB(self):
        return (randint(1, 255), randint(1, 255), randint(1, 255))

    def boundaries(self, cnt):
        lm = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rm = tuple(cnt[cnt[:, :, 0].argmax()][0])
        tm = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bm = tuple(cnt[cnt[:, :, 1].argmax()][0])
        return (lm, rm, tm, bm)

    def axes(self, arr, func):
        x = func([i[0] for i in arr])
        y = func([i[1] for i in arr])
        return (x, y)

    def cropAndResize(self):
        temp = self.opt_boxes[0]
        # combine into a single array
        for i in self.opt_boxes[1:]:
            temp = np.vstack([temp, i])
        # find boundaries
        res = self.boundaries(temp)
        # find axes(x and y points)
        ax = []
        for i in (min, max):
            ax.append(self.axes(res, i))

        minX = ax[0][0]
        minY = ax[0][1]
        maxX = ax[1][0]
        maxY = ax[1][1]
        # crop
        crop_img = self.img[minY:maxY, minX:maxX].copy()
        # resize
        resized = cv2.resize(crop_img, None, fx=0.8, fy=0.8)

    def getOptionLetters(self):
        if self.dataset == None:
            return "No dataset selected"
        contours = self.getOptionLetterContours()
        letters = []
        for cnt in contours:
            ltr = ''
            temp = 1
            for letter, value in self.dataset.items():
                ret = cv2.matchShapes(cnt[0], value[0], 1, 0.0)
                if ret < 0.1 and ret < temp:
                    temp = ret
                    ltr = letter
                    # print(ret, ltr)
            letters.append(ltr)
            # if '' in letters:
            #     im_rgb = cv2.imread(self.path)
            #     gray = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY)
            #     template = cv2.imread("C.png", 0)
            #     w, h = template.shape[::-1]
            #     # im, cntc, hier = cv2.findContours(
            #     #     bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     # TM_SQDIFF and #TM_CCORR_NORMED
            #     res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            #     threshold = 0.8
            #     loc = np.where(res >= threshold)
            #     for pt in zip(*loc[::-1]):
            #         cv2.rectangle(
            #             im_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

            # cv2.imwrite('res.png', im_rgb)
        if '' in letters:
            letters[letters.index('')] = 'C'
        return letters

    def exec(self):
        self.readImage()
        self.drawPlane()
        self.thresholdImage()
        self.findContours()
        self.separateContours()
        self.writeIntoPlane()


class Dataword:
    def __init__(self, letters, opt_pts, ans_pts, words):
        self.letters = list(map(str.lower, letters))
        self.opt_pts = opt_pts
        self.ans_pts = ans_pts
        self.opt_len = len(opt_pts)
        self.ans_len = len(ans_pts)
        self.words = words
        self.preWords = None
        self.combos = []
        self.validCombos = []
        if self.opt_len != len(letters):
            raise Exception("Number of Letters != Number of options")

    def findCombinations(self):
        self.combos = list(permutations(
            range(len(self.letters)), self.ans_len))

    def getValidWords(self):
        for c in self.combos:
            word = ''.join([self.letters[i] for i in c])
            if word in self.words and word not in self.preWords:
                print(word)
                self.validCombos.append(c)
                self.preWords[word] = 1
                json.dump(self.preWords, open("../json/preWords.json", "w"))
                break

    def getPoints(self):
        for c in self.validCombos:
            return [self.opt_pts[i] for i in c]

    def readPreWordList(self):
        self.preWords = json.load(open("../json/preWords.json", "r"))

    def exec(self):
        self.findCombinations()
        self.readPreWordList()
        self.getValidWords()


def getScreenshot():
    global cmd
    capture = cmd[0]
    os.system(capture)
    pull = cmd[1]
    os.system(pull)


def processImage():
    # fix "C" and 588/160
    dataset = readDataSet()
    objects = Objects(dataset=dataset)
    objects.exec()
    letters = objects.getOptionLetters()
    ans_btns = objects.getAnswerButton()
    opt_btns = objects.getOptionButton()

    if letters == [] or ans_btns == [] or opt_btns == []:
        return None
    return letters, ans_btns, opt_btns


def processData(letters, ans_btns, opt_btns, words):
    dword = Dataword(letters, opt_btns, ans_btns, words)
    dword.exec()
    pts = dword.getPoints()

    return pts


def sendInputs(pts):
    global cmd
    for pt in pts:
        os.system(cmd[2]+" "+str(pt[0])+" "+str(pt[1]))


def main1():
    words = readWordList()

    def wait():
        global cmd
        os.system(cmd[3])
        time.sleep(1)

    while(1):
        try:
            try:
                getScreenshot()
                ltrs, a_btn, o_btn = processImage()
                pts = processData(ltrs, a_btn, o_btn, words)
                sendInputs(pts)
                time.sleep(2)
            except IndexError:
                wait()
            except TypeError:
                wait()
            except ZeroDivisionError:
                wait()
        except KeyboardInterrupt:
            choice = input("Pause/Stop (P/S)?").lower()
            if choice == 'p':
                cont = input("Press enter to continue")
                continue
            else:
                break


if __name__ == "__main__":
    main1()
