import os
import cv2
import numpy as np
import math
import os
import json
import zlib
import base64

def getChannelNum(img):
    if img.ndim == 2:
        return 1 #单通道
    else:
        return img.shape[2]

def premultOp(arr,ratio):
    ret = np.uint8(np.ceil(np.multiply(arr, ratio)))
    return ret

def premultAlpha(img):
    num = getChannelNum(img)
    if num == 4:
        b,g,r,a = cv2.split(img)
        ratio=a / 255.0
        img = cv2.merge((premultOp(b,ratio),premultOp(g,ratio),premultOp(r,ratio)))
    elif num == 1:
        img = cv2.merge((img,img,img))
    return img

def compressHash(hashArr):
    compressed_data = zlib.compress(hashArr)
    return base64.b64encode(compressed_data).decode('utf-8')

def decompressHash(hashStr):
    decompressed_bytes = zlib.decompress(base64.b64decode(hashStr))
    decompressed_arr = np.frombuffer(decompressed_bytes, dtype=np.uint8)
    return decompressed_arr


def bytesToKey(hashArr):
    compressed_data = zlib.compress(hashArr)
    return base64.b64encode(compressed_data).decode('utf-8')

def keyToBytes(hashStr):
    decompressed_bytes = zlib.decompress(base64.b64decode(hashStr))
    decompressed_arr = np.frombuffer(decompressed_bytes, dtype=np.uint8)
    return decompressed_arr
    
def trim(img):
    h = img.shape[0]
    w = img.shape[1]
    topRow = 0
    bottomRow = h
    leftCol = 0
    rightCol = w

    for i in range(0,h):
        gray=cv2.cvtColor(premultAlpha(img[i:i+1]), cv2.COLOR_BGR2GRAY)
        if np.mean(gray) == 0:
            topRow = i+1
        else:
            break

    for i in range(h,0,-1):
        gray=cv2.cvtColor(premultAlpha(img[i-1:i]), cv2.COLOR_BGR2GRAY)
        if np.mean(gray) == 0:
            bottomRow = i-1
        else:
            break

    for i in range(0,w):
        gray=cv2.cvtColor(premultAlpha(img[:,i:i+1]), cv2.COLOR_BGR2GRAY)
        if np.mean(gray) == 0:
            leftCol = i+1
        else:
            break

    for i in range(w,0,-1):
        gray=cv2.cvtColor(premultAlpha(img[:,i-1:i]), cv2.COLOR_BGR2GRAY)
        if np.mean(gray) == 0:
            rightCol = i-1
        else:
            break

    # print("shape",img.shape,"trim:",topRow,bottomRow,leftCol,rightCol)
    return img[topRow:bottomRow,leftCol:rightCol]

class SimilarPicFinder(object):
    dir = ''
    def __init__(self, dir,dataFile):
        self.dir = dir  # 实例属性
        self.photoData = {}
        self.photoHash = {}
        self.bucket = {}
        self.photoData["photoHash"] = self.photoHash
        self.photoData["bucket"] = self.bucket
        self.shape=(16,16)
        self.rootHash = bytesToKey(np.uint8(np.ones(self.shape[0]*self.shape[1])))
        self.distance = 20
        self.hashFile = dataFile
 
    #pHash优化，
    def pHashEx(self,img):
        if np.size(img) == 0:
            return None
        img = cv2.resize(img, (64,64),interpolation=cv2.INTER_AREA)
        img = premultAlpha(img)
        # 转换为灰度图
        yrb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(yrb)
        # 将灰度图转为浮点型，再进行dct变换
        dct = cv2.dct(np.float32(y))
        #截取左上角低频部分
        dct_roi = dct[0:self.shape[0], 0:self.shape[1]]
        img_back = cv2.idct(dct_roi)
        minValue = np.min(img_back)
        if minValue < 0:
            img_back = np.add(img_back,abs(minValue))
        if minValue == 0 and np.max(img_back) ==0:
            return None
        median = np.divide(img_back,np.median(img_back))
        normalize = np.divide(median,np.max(median))
        byteArr = np.floor(np.multiply(normalize,255))
        return bytesToKey(np.uint8(byteArr).flatten())
 
    # 感知哈希算法(pHash)
    def pHash(self,img):
        img = cv2.resize(img, (64,64),interpolation=cv2.INTER_AREA)
        img = premultAlpha(img)

        # 转换为灰度图
        yrb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(yrb)

        # 将灰度图转为浮点型，再进行dct变换
        dct = cv2.dct(np.float32(y))
        # opencv实现的掩码操作
        dct_roi = dct[0:self.shape[0], 0:self.shape[1]]

        md = 2
        means = np.zeros((int(self.shape[0]/md), int(self.shape[1]/md)))
        for i in range(md):
            for j in range(md):
                block = dct_roi[i*md:(i+1)*md, j*md:(j+1)*md]
                average = np.mean(block)
                means[i][j] = average
 
        hash_str = ''
        for i in range(dct_roi.shape[0]):
            for j in range(dct_roi.shape[1]):
                mi = int(i / md)
                mj = int(j / md)
                if dct_roi[i, j] > means[mi][mj]:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        return hash_str
 
    # Hash值对比:
    def cmpHash(self,hash1,hash2):
        vector1 = keyToBytes(hash1)
        vector2 = keyToBytes(hash2)
        # 归一化向量
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        #计算余弦值[1,-1]
        similarity = np.dot(vector1, vector2)
        return similarity

 
    # 定义函数
    def list_all_files(self,rootdir):
        _files = []
        # 列出文件夹下所有的目录与文件
        list = os.listdir(rootdir)
        for i in range(0, len(list)):
            # 构造路径
            path = os.path.join(rootdir, list[i])
            # 判断路径是否为文件目录或者文件
            # 如果是目录则继续递归
            if os.path.isdir(path) and not list[i].startswith('.'):
                _files.extend(self.list_all_files(path))
            if os.path.isfile(path):
                _files.append(path)
        return _files
    
    def computeHash(self,photo):
        if(not os.path.isfile(photo)):
            return None
        fpath,fname=os.path.split(photo)
        ffname = fname.split('.')
        
        if len(ffname) < 2:
            return None

        #不是下列文件形式跳出
        if(ffname[1] not in {'jpg', 'bmp', 'png', 'jpeg'}):
            return None
        
        print('Handle:',fname)
        img = cv2.imdecode(np.fromfile(photo,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Error decode image:",photo)
            return None
        hash = self.pHashEx(trim(img))
        if hash is None:
            print("Empty Hash:",photo) #遮罩图暂不处理
            return None
        
        return hash,fname
 
    #处理文件
    def buildHash(self):
        photoList = self.list_all_files(self.dir)
        for i,photo in enumerate(photoList):
            result = self.computeHash(photo)
            if result is None:
                continue

            hash,fname = result
            if self.photoHash.get(hash) is None:
                self.photoHash[hash] = photo

            if hash != '':
                self.addBucket(hash)

        with open(self.hashFile, "w") as file:
            json.dump(self.photoData, file)

    def addBucket(self,hash):
        index = str(round(self.cmpHash(self.rootHash,hash) * 1000)) #转成[-1000,1000]
        if self.bucket.get(index) is None:
            self.bucket[index] = []
        self.bucket[index].append(hash)
        return index

    def findBucket(self,hash,fname):
        if hash ==  '':
            return None
        maxSim = 0
        maxHash = ''
        index = str(round(self.cmpHash(self.rootHash,hash) * 1000))
        if self.bucket.get(index) is None:
            return None
        
        count = 0
        for i in range(-self.distance,self.distance+1):
            curIndex = int(index)+i
            if curIndex >=-1000 and curIndex <= 1000:
                hashList = self.bucket.get(str(curIndex)) or []
                for j,targetHash in enumerate(hashList):
                    similar = self.cmpHash(hash,targetHash)
                    count = count + 1
                    if similar > maxSim:
                        maxSim = similar
                        maxHash = targetHash
        if maxSim > 0:
            print("find similar:",maxSim,self.photoHash[maxHash],fname,count)
            return self.photoHash[maxHash]
        else:
            print("not find")
            return None
                
    def loadHash(self):
        if os.path.exists(self.hashFile):
            print("load json")
            with open(self.hashFile, "r") as file:
                self.photoData = json.load(file)
                self.photoHash = self.photoData["photoHash"]
                self.bucket = self.photoData["bucket"]
                return True
        return False

    def findPics(self,targetDir):
        photoList = self.list_all_files(targetDir)
        for i,photo in enumerate(photoList):
            result = self.computeHash(photo)
            if result is None:
                continue

            hash,fname = result
            if self.photoHash.get(hash) is not None:
                print("find same:",self.photoHash[hash],fname)
            else:
                self.findBucket(hash,fname)

    def findPic(self,file):
        result = self.computeHash(file)
        if result is not None:
            hash,fname = result
            if self.photoHash.get(hash) is not None:
                return self.photoHash[hash]
            else:
                return self.findBucket(hash,fname)
        return None

