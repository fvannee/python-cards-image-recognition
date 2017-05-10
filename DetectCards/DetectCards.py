import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import hashlib
import glob
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

import colormath.color_diff

from tkinter import Tk

CARD_SET = 'cartamundi' # grimaud or cartamundi
DATA_DIR = CARD_SET + '/data/'
CARDS_DIR = CARD_SET + '/cards/'

def plot_clipboard():
   r = Tk()
   cp = r.selection_get(selection = "CLIPBOARD")
   plot_maximized(to_array(cp))

def to_array(s):
   return np.array(np.matrix(s.strip('[]')))

def angle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product/(len1*len2))

def angle_p(x, y):
   rads = math.atan2(-y,x)
   rads %= 2*math.pi
   return rads

def plot_maximized(img, gray=True):
   plt.imshow(img, 'gray' if gray else None)
   mng = plt.get_current_fig_manager()
   mng.window.state('zoomed')
   plt.show()

def load_and_convert(img_path):
   img_color = cv2.imread(img_path, -1)
   coefficients = [0.5,1.5,-1]
   #coefficients = [0.7,2.0,-1]
   # for standard gray conversion, coefficients = [0.114, 0.587, 0.299]
   m = np.array(coefficients).reshape((1,3))
   img1 = cv2.transform(img_color, m)
   #use otsu threshold
   #thresh, img2 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   #or use adaptive threshold
   img2 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
         cv2.THRESH_BINARY,101,40)
   
   return img2, img_color

def show_black_white(img_path, threshold=140, gray=False):
   img1 = cv2.imread(img_path, -1)
   #coefficients = [0.114-0.05, 0.587-0.1, 0.299+0.15]
   coefficients = [0.5,1.5,-1]
   # for standard gray conversion, coefficients = [0.114, 0.587, 0.299]
   m = np.array(coefficients).reshape((1,3))
   img1 = cv2.transform(img1, m)
   
   #(thresh, img1) = cv2.threshold(img1, threshold, 255, cv2.THRESH_BINARY)
   # Otsu's thresholding after Gaussian filtering
   #plot_maximized(img1)
   #blur = cv2.GaussianBlur(img1,(5,5),0)
   #plot_maximized(blur)
   thresh, img2 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   #plot_maximized(img2)

   th2 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
         cv2.THRESH_BINARY,101,40)
   #th3 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
   #      cv2.THRESH_BINARY,101,2)
   plot_maximized(th2)
   #plot_maximized(th3)
   #(thresh, img1) = cv2.threshold(img1, threshold, 255, cv2.THRESH_OTSU)

#for img in IMGS2:
#   show_black_white(img, 120, True)

def proper_subimage(image, contour, center, theta, width, height):
   mask = np.zeros(image.shape, np.uint8)
   mask = cv2.fillPoly(mask, contour, 255)
   inv_mask = cv2.bitwise_not(mask)
   masked_img = cv2.bitwise_and(image, mask)
   cv2.bitwise_or(masked_img, inv_mask, masked_img)

   theta *= math.pi / 180 # convert to rad
   
   v_x = (math.cos(theta), math.sin(theta))
   v_y = (-math.sin(theta), math.cos(theta))
   s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
   s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

   mapping = np.array([[v_x[0],v_y[0], s_x],
                  [v_x[1],v_y[1], s_y]])
   return cv2.warpAffine(masked_img,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)


def subimage(image, center, theta, width, height):
   theta *= math.pi / 180 # convert to rad
   
   v_x = (math.cos(theta), math.sin(theta))
   v_y = (-math.sin(theta), math.cos(theta))
   s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
   s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

   mapping = np.array([[v_x[0],v_y[0], s_x],
                  [v_x[1],v_y[1], s_y]])
   return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

SMALL_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50]

def draw_rectangle_box(img, rect):
   box = cv2.boxPoints(rect)
   box = np.int0(box)
   cv2.drawContours(orig_img,[box],0,(0,0,255),3)

def rot90(rect, k=1):
   w = rect[1][0] if k % 2 == 0 else rect[1][1]
   h = rect[1][1] if k % 2 == 0 else rect[1][0]
   return (rect[0], (w, h), rect[2] + k*90)

def cards_to_set(cards):
   s = set()
   for card in cards:
      s.add(card[0])
   return s

def is_red_or_black(pixel):
   rgb = sRGBColor(pixel[2], pixel[1], pixel[0], True)
   xyz = convert_color(rgb, LabColor)

   blue, green, red = pixel[0], pixel[1], pixel[2]
   colors_names = ['r', 'b', 'r']
   colors = [sRGBColor(255,0,0,True), sRGBColor(0,0,0,True), sRGBColor(255,128,0,True)]
   colors = [convert_color(c, LabColor) for c in colors]

   best_match = -1
   best_value = 1e9
   for i in range(len(colors)):
      dist = np.linalg.norm(np.array(xyz.get_value_tuple()) - np.array(colors[i].get_value_tuple()))
      dist = colormath.color_diff.delta_e_cie2000(xyz, colors[i])
      #dist = colour_distance(pixel, colors[i])
      if dist < best_value:
         best_value = dist
         best_match = i
   return colors_names[best_match]
   r_to_b = red / (blue+1)
   r_to_g = red / (green+1)
   if red > 30 and r_to_b > 3 and r_to_g > 2.5:
      return 'r'
   if red > 65 and r_to_b > 2.3 and r_to_g > 1.8:
      return 'r'
   if red > 100 and r_to_b > 1.5 and r_to_g > 1.3:
      return 'r'
   elif red > 130 and (r_to_b > 1.3 or r_to_g > 1.3):
      return 'r'
   elif red > 150 and (r_to_b > 1.15 or r_to_g > 1.15):
      return 'r'
   else:
      return 'b'

def colour_distance(e1, e2):
   blue1, green1, red1 = e1[0], e1[1], e1[2]
   blue2, green2, red2 = e2[0], e2[1], e2[2]
   rmean = ( red1 + red2 ) / 2
   r = red1 - red2
   g = green1 - green2
   b = blue1 - blue2
   return math.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256))


INTERESTING_AREA_PARAM = 6000
MIN_DIM = 3.5
MIN_SUIT_DIM = 2
CARDS = ['0', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['C', 'D', 'H', 'S']

def is_interesting_rectangle(img, rect):
   width = len(img)
   height = len(img[0])
   min_area = width * height / INTERESTING_AREA_PARAM
   w, h = rect[1]
   dim = w / h if h != 0 else 0
   return w * h > min_area and dim > (1/MIN_DIM) and dim < MIN_DIM

MATCH_FOUND_PARAM = 7000
TIGHT_MATCH_FOUND_PARAM = 6000

class CardDetection(object):
   def __init__(self, samples, responses):
      
      self.train(samples, responses)
      #self._model = cv2.ml.KNearest_create()
      #if samples.size > 0:
      #   self._model.train(samples, cv2.ml.ROW_SAMPLE, responses)
      
   @staticmethod
   def _split_samples_and_responses(samples, responses):
      black_samples = np.empty((0, samples.shape[1]), dtype=np.float32)
      red_samples = np.empty((0, samples.shape[1]), dtype=np.float32)
      black_responses = np.empty((0, 1), dtype=np.float32)
      red_responses = np.empty((0, 1), dtype=np.float32)
      for sample, response in zip(samples, responses):
         sample = sample.reshape((1, len(sample)))
         response_char = chr(response[0])
         response = response.reshape((1, len(sample)))
         if response_char != 'D' and response_char != 'H':
            black_samples = np.append(black_samples, sample, 0)
            black_responses = np.append(black_responses, response, 0)
         if response_char != 'S' and response_char != 'C':
            red_samples = np.append(red_samples, sample, 0)
            red_responses = np.append(red_responses, response, 0)
      return ((red_samples, red_responses), (black_samples, black_responses))

   @staticmethod
   def from_file(sample_file, response_file):
      samples = np.loadtxt(sample_file, np.float32)
      responses = np.loadtxt(response_file, np.float32)
      responses = responses.reshape((responses.size,1))
      return CardDetection(samples, responses)

   def train(self, samples, responses):
      ((self.red_samples, self.red_responses), (self.black_samples, self.black_responses)) = \
         self._split_samples_and_responses(samples, responses)
      self._small_size = int(math.sqrt(samples.shape[1]))
      self._black_model = cv2.ml.KNearest_create()
      self._red_model = cv2.ml.KNearest_create()
      if len(self.red_samples) > 0:
         self._red_model.train(self.red_samples, cv2.ml.ROW_SAMPLE, self.red_responses)
      if len(self.black_samples) > 0:
         self._black_model.train(self.black_samples, cv2.ml.ROW_SAMPLE, self.black_responses)

   def is_trained(self, color):
      if color == 'b':
         return self._black_model.isTrained()
      else:
         return self._red_model.isTrained()

   def is_suit_or_rank_from_sub_img(self, sub_img, color, required_accuracy=MATCH_FOUND_PARAM):
      dim = sub_img.shape[0] / sub_img.shape[1]
      suits = ['S', 'H', 'D', 'C']
      roismall_orig = cv2.resize(sub_img,(self._small_size, self._small_size))
      b_retval, b_results, b_neigh_resp, b_dists = ([],[],[],[[1e9]])
      is_rotated = False
      for i in range(2):
         roismall = roismall_orig.reshape((1, self._small_size * self._small_size))
         roismall = np.float32(roismall)
         if color == 'r':
            retval, results, neigh_resp, dists = self._red_model.findNearest(roismall, k = 1)
         elif color == 'b':
            retval, results, neigh_resp, dists = self._black_model.findNearest(roismall, k = 1)
         else:
            raise Exception('Invalid input color')
         #print(dists[0][0], results[0][0])
         if dists[0][0] < b_dists[0][0]:
            b_retval, b_results, b_neigh_resp, b_dists = retval, results, neigh_resp, dists
            is_rotated = i == 1
         if i != 1:
            roismall_orig = np.rot90(roismall_orig, 2)
      match_found = b_dists[0][0] < required_accuracy * (self._small_size * self._small_size) and \
         (chr(int((b_results[0][0]))) not in suits or dim < MIN_SUIT_DIM or 1/dim > MIN_SUIT_DIM)
      #print('best match', b_dists[0][0], chr(b_results[0][0]), match_found)
      if not match_found:
         return (None, None)
      else:
         string = chr(int((b_results[0][0])))
         if string in suits:
            if string == 'S' and color == 'r':
               raise Exception('not possible')
            if string == 'H' and color == 'b':
               raise Exception('not possible')
            if string == 'D' and color == 'b':
               raise Exception('not possible')
         return (string, is_rotated)

   def is_suit_or_rank_from_contour(self, img, color_img, contour, debug_img=None):
      rect = cv2.minAreaRect(contour)
      w, h = rect[1]
      ans = (None, None)
      if is_interesting_rectangle(img, rect):
         if w > h:
            rect = rot90(rect)
            w, h = rect[1]
         if debug_img is not None:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(debug_img, [box], 0, (0,0,255), 3)
         #roi = subimage(img, rect[0], rect[2], int(w), int(h))
         roi = subimage(img, rect[0], rect[2], int(w), int(h))
         color = is_red_or_black(color_img[int(rect[0][1])][int(rect[0][0])])
         feature, rotated = self.is_suit_or_rank_from_sub_img(roi, color)
         if feature is not None:
            if rotated:
               rect = (rect[0], (h, w), (rect[2] + 180) % 360)
            ans = (rect, feature)
         if debug_img is not None:
            cv2.putText(debug_img, feature if feature is not None else '-', (int(rect[0][0]), int(rect[0][1])), 0, 2, (255,255,0), 5, cv2.LINE_AA)
      return ans

   def collect_features_from_image(self, img, color_img, debug_img=None):
      results = []
      flags, contours, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      for cnt in contours:
         rect, feature = self.is_suit_or_rank_from_contour(img, color_img, cnt, debug_img)
         if feature != None:
            results.append((feature, rect))
      return results

   def detect_cards_in_image(self, img, color_img, debug):
      debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if debug else None
      features = self.collect_features_from_image(img, color_img, debug_img)
      cards = self.features_to_cards(features, debug_img)
      return cards, debug_img

   def features_to_cards(self, features, debug_img=None):
      results = []
      for (i, (s, ((x,y),(w,h),r))) in enumerate(features):
         if s in CARDS:
            m = max(w,h) * max(w,h)
            (best_card, (best_x, best_y), best_angle) = (None, (0,0), 180)
            for (s2,((x2,y2),_,_)) in features:
               if s2 in SUITS:
                  if (x2 - x) * (x2 - x) + (y2 - y) * (y2 - y) < 1.3 * m:
                     line_angle = angle_p(x2 - x, y - y2) / math.pi * 180
                     angle_diff = ((line_angle + 360 - r) % 360)
                     ANGLE_ERROR = 35
                     angle_error = min(math.fabs(angle_diff - 90), math.fabs(angle_diff - 270))
                     #if (angle_diff > (90 - ANGLE_ERROR) and angle_diff < (90 + ANGLE_ERROR)) or \
                     #   (angle_diff > (270 - ANGLE_ERROR) and angle_diff < (270 + ANGLE_ERROR)):
                     if angle_error < ANGLE_ERROR and angle_error < best_angle:
                        if s == '9':
                           if angle_diff > 180:
                              s = '6'
                        (best_card, (best_x, best_y), best_angle) = (s2 + s, (x2, y2), angle_error)
            if best_card is not None:
               results.append((best_card, (best_x, best_y)))
               if debug_img is not None:
                  #cv2.putText(debug_img, s2 + s + (' %.1f %.1f' % (line_angle, r)),(int(x2),int(y2)),0,1.5,(0,255,75),5, cv2.LINE_AA)
                  cv2.putText(debug_img, best_card, (int(best_x), int(best_y)), 0, 2, (0,255,75), 5, cv2.LINE_AA)
      return results

OK = 0
EXIT = 1
PREVIOUS = 2
KEYS = [i for i in range(50,58)] + [ord(c) for c in ['T', 'J', 'Q', 'K', 'A', 'S', 'H', 'D', 'C']]

class CardTraining(object):
   def __init__(self, sizes, samples=None, responses=None, ignores=None, detect_size=25):
      self._sizes = sizes
      self._detect_size = detect_size
      self._detect_size_i = self._sizes.index(detect_size)
      if samples is None:
         self._samples = [np.empty((0, size * size)) for size in sizes]
      else:
         self._samples = samples
      if responses is None:
         self._responses = [[] for _ in sizes]
      else:
         self._responses = responses
      if ignores is None:
         self._ignores = []
      else:
         self._ignores = ignores
      resp = np.array(self._responses[self._detect_size_i], np.float32)
      resp = resp.reshape((resp.size,1))
      self._detect = CardDetection(self._samples[self._detect_size_i], resp)

   def ask_rectangle(self, img, rect, color_img, contour=None):
      flip = False
      w, h = rect[1]
      if w > h:
         rect = rot90(rect)
         w, h = rect[1]
      cont = True
      while cont:
         #if flip:
         #   rect = rot90(rect, 2)
         box = cv2.boxPoints(rect)
         box = np.int0(box)
         #cv2.drawContours(orig_img,[box],0,(0,0,255),3)
         #roi = subimage(img, rect[0], rect[2], int(w), int(h))
         roi = subimage(img, rect[0], rect[2], int(w), int(h))
         if flip:
            roi = np.rot90(roi, k=2)
         roi_hash = hashlib.sha1(roi.data.tobytes()).digest()
         possibleIgnore = True
         color = is_red_or_black(color_img[int(rect[0][1])][int(rect[0][0])])
         if self._detect.is_trained(color):
            feature, rotated = self._detect.is_suit_or_rank_from_sub_img(roi, color, MATCH_FOUND_PARAM)
            if feature is not None:
               #dbg_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
               #cv2.putText(dbg_roi, feature,(int(0),int(len(dbg_roi)/2)),0,2,(0,255,75),5, cv2.LINE_AA)
               #cv2.imshow('train', dbg_roi)
               #key = cv2.waitKey(0)
               print('known value %s ' % feature)
               if feature != 'H' or True:
                  return (roi, None, None, OK)
               else:
                  possibleIgnore = False
         if roi_hash in self._ignores and possibleIgnore:
            return (roi, None, None, OK)
         cv2.imshow('train', roi)
         key = cv2.waitKey(0)
         key = ord(chr(key).upper())
         if key == 61:
            flip = not flip
         elif key == 27:  # (escape to quit)
            return (None, None, None, EXIT)
         elif (key == ord('S') or key == ord('C')) and color == 'r':
            print('Cannot add S/C when color is red')
            print(rect, color_img[int(rect[0][1])][int(rect[0][0])])
         elif (key == ord('D') or key == ord('H')) and color == 'b':
            print('Cannot add D/H when color is black')
            print(rect, color_img[int(rect[0][1])][int(rect[0][0])])
         elif key == ord('6'):
            print('Did you mean 9? Please try again')
         elif key == ord('0'):
            print('Did you mean T? Please try again')
         elif key in KEYS:
            return (roi, key, flip, OK)
         elif key == 8:
            return (None, None, None, PREVIOUS)
         else:
            self._ignores.append(roi_hash)
            return (roi, None, None, OK)

   def train_image(self, img, color_img):
      flags, contours, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      for cnt in contours:
         rect = cv2.minAreaRect(cnt)
         if is_interesting_rectangle(img, rect):
            (roi, key, flipped, code) = self.ask_rectangle(img, rect, color_img, cnt)
            if code == OK:
               if key is not None:
                  for i, size in enumerate(self._sizes):
                     self._responses[i].append(key)
                     roismall = cv2.resize(roi, (size, size))
                     sample = roismall.reshape((1, size * size))
                     self._samples[i] = np.append(self._samples[i], sample, 0)
                     if size == self._detect_size:
                        train_responses = np.array(self._responses[i], np.float32)
                        train_responses = train_responses.reshape((train_responses.size,1))
                        self._detect.train(np.float32(self._samples[i]), train_responses)
            elif code == EXIT:
               return False
      return True

   def write_to_files(self, sample_files, responses_files, ignore_file):
      for i, size in enumerate(self._sizes):
         to_write = np.array(self._responses[i], np.float32)
         to_write = to_write.reshape((to_write.size,1))

         np.savetxt(sample_files % size, self._samples[i])
         np.savetxt(responses_files % size, to_write)
      with open(ignore_file, 'wb') as f:
         pickle.dump(self._ignores, f)

   @staticmethod
   def from_files(sizes, sample_files, responses_files, ignore_file):
      samples = []
      responses = []
      ignores = []
      for size in sizes:
         f_samples = sample_files % size
         f_responses = responses_files % size
         if os.path.isfile(f_samples) and os.path.isfile(f_responses):
            samples.append(np.loadtxt(f_samples, np.float32))
            responses.append(np.loadtxt(f_responses, np.float32).tolist())
         else:
            samples.append(np.empty((0, size * size), dtype=np.float32))
            responses.append([])
      if os.path.isfile(ignore_file):
         with open(ignore_file, 'rb') as f:
            ignores = pickle.load(f)
      return CardTraining(sizes, samples, responses, ignores)

def test(imgs, small_size=25):
   detect = CardDetection.from_file(DATA_DIR + 'samples_%d.data' % small_size, DATA_DIR + 'responses_%d.data' % small_size)
   for img_path in imgs:
      img, color_img = load_and_convert(img_path)
      cards, debug_img = detect.detect_cards_in_image(img, color_img, True)
      debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
      plot_maximized(debug_img, False)

def show_samples(inp, small_size=25):
   detect = CardDetection.from_file(DATA_DIR + 'samples_%d.data' % small_size, DATA_DIR + 'responses_%d.data' % small_size)
   for (i, (sample, response)) in enumerate(zip(detect._samples, detect._responses)):
      if response == ord(inp):
         print(sample, i+1)
         sample = sample.reshape((small_size, small_size))
         plt.imshow(sample, 'gray')
         plt.show()

def train_all(imgs, sizes, reuse=True, save=True):
   sample_files = DATA_DIR + 'samples_%d.data'
   responses_files = DATA_DIR + 'responses_%d.data'
   ignore_file = DATA_DIR + 'ignores.data'
   if reuse:
      train = CardTraining.from_files(sizes, sample_files, responses_files, ignore_file)
   else:
      train = CardTraining(sizes)
   for img_path in imgs:
      print(img_path)
      img, color_img = load_and_convert(img_path)
      if not train.train_image(img, color_img):
         break
   if save:
      train.write_to_files(sample_files, responses_files, ignore_file)


      
#IMG = r'IMG_20170312_153128.jpg'
#IMG='hearts2.png'
#IMG2 = r'IMG_20170312_153950.jpg'
#IMGS = ['cards/' + f for f in ['tjqka.jpg', 'suits.jpg', '2367.jpg', '4589.jpg']]
#IMGS = [CARDS_DIR + 'IMG_20170318_183641.jpg'] + [CARDS_DIR + f for f in ['hearts.png', 'clubs.png','diamonds.png', 'spades.png']]
#IMGS2 = ['IMG_20170312_153128.jpg'] + [CARDS_DIR + f for f in ['2367-2.jpg', 'tjqka-2.jpg', 'suits-2.jpg', '4589-2.jpg']] + ['IMG_20170312_153950.jpg']
#IMGS2 = [CARDS_DIR + f for f in ['IMG_20170319_213511.jpg', '2367.jpg', 'IMG_20170319_222544.jpg', 'IMG_20170319_222533.jpg', 'IMG_20170319_220259.jpg', 'IMG_20170319_220254.jpg', 'IMG_20170319_220247.jpg', 'IMG_20170319_213511.jpg', 'IMG_20170319_213451.jpg', 'IMG_20170319_213238.jpg', 'IMG_20170319_213034_1.jpg']]
IMGS2 = glob.glob(CARDS_DIR + '*.jpg')
IMGS2.reverse()
#IMGS2=IMGS

print(IMGS2)
train_all(IMGS2, SMALL_SIZES, True, False)
test(IMGS2)
#show_samples('H')

