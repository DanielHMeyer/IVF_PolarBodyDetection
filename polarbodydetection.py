# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from skimage import feature, transform, io, draw
from sklearn.cluster import DBSCAN
import os

def load_images(path, color):
    os.chdir(path)
    filenames = [f for f in os.listdir('.') if os.path.isfile(f)]
    images = []
    for file in filenames:
        images.append(io.imread(file, not color))
    return images

# TODO: Add docstring
class PolarBodyDetector:
    """
    A class that can detect the presence of a polar body
    
    Args:
        
    Returns:
        
    """
    def __init__(self, pipTemplate, startImg, cSigma=2, cLow=0.1, cHigh=0.6, minR=60,
                 eSigma=2, eThres=0.1, dbEps=20, dbSamples=150):
        self.pipTemplate = pipTemplate      # Template of the pipette tip
        self.startImg = startImg            # First image before detection
        self.cSigma = cSigma                # sigma of gaussian filter for canny edge detection
        self.cLow = cLow                    # low threshold for canny
        self.cHigh = cHigh                  # high threshold for canny
        self.minR = minR                    # minimum radius of oolemma
        self.eSigma = eSigma                # sigma of gaussian filter for edge detection
        self.eThres = eThres                # threshold for edge detection
        self.dbEps = dbEps                  # distance of samples for clustering
        self.dbSamples = dbSamples          # minimum number of samples in a cluster
        self._detect_oocyte_roi(startImg)
        self._detect_oolemma_roi(startImg)
        
    def _detect_oocyte_roi(self, img):
        res = feature.match_template(img, self.pipTemplate)
        ij = np.unravel_index(np.argmax(res), res.shape)
        self.coordROI = (ij[0]-45, ij[1]-200, 250, 200)
        roi = img[ij[0]-45:ij[0]+205,ij[1]-200:ij[1]]
        return roi
        
    def _detect_oolemma_roi(self, img):
        y, x , h, w = self.coordROI
        roi = img[y:y+h, x:x+w]
        # find canny edges to detect oolemma
        roiC = feature.canny(roi, sigma=self.cSigma,
                             low_threshold=roi.mean()*self.cLow, 
                             high_threshold=roi.mean()*self.cHigh)
        
        # Create a bounding box and find center of box
        lr = roiC.mean(axis=0)
        ud = roiC.mean(axis=1)
        l, r = np.nonzero(lr)[0][0], np.nonzero(lr)[0][-1]
        u, d = np.nonzero(ud)[0][0], np.nonzero(ud)[0][-1]
        cy, cx = int((u+d)/2.0), int((r+l)/2.0)
             
        dist = np.asarray([(r-l)/2.0, (d-u)/2.0])
        
        self.minDist = dist.min()
        self.maxDist = dist.max()
        aveDist = (self.maxDist+self.minDist)/2
        self.coordOO = (cy, cx, aveDist)
        
    def _create_patch(self, img):
        cy, cx, aveDist = self.coordOO
        y, x , h, w = self.coordROI
        roi = img[y:y+h, x:x+w]
        
        minDist = int(aveDist-10)
        maxDist = int(aveDist+30)
        
        # Extract a patch by 
        imPad = np.zeros((400,400), dtype=np.float64)
        imPad[200-cy:450-cy, 200-cx:400-cx] = roi
        imRot = transform.rotate(imPad, angle=30, center=[200,200])
        patch = np.expand_dims(imRot[200-maxDist:200-minDist, 200],axis=1)
        for i in range(29,-210,-1):
            imRot = transform.rotate(imPad, angle=i, center=[200,200])
            vert = np.expand_dims(imRot[200-maxDist:200-minDist, 200],axis=1)
            patch = np.hstack([patch, vert])
        return patch, (cy,cx)
            
        
    def _detect_polar_body_patch(self, patch):
        # Find edges in patch
        patchKT = feature.corner_shi_tomasi(patch, sigma=self.eSigma)
        patchKT = (patchKT-patchKT.min())/(patchKT.max()-patchKT.min())
        patchTH = patchKT.copy()
        # threshold edges
        th = self.eThres
        patchTH[patchKT<=th] = 0
        patchTH[patchKT>th] = 255
        patchTH[:,0:6] = 0
        patchTH[:,235:] = 0
        
        # extract coordinates of keypoints
        keyps = np.where(patchTH==255)
        keyps = np.asarray([[y, x] for (y,x) in zip(keyps[0], keyps[1])])
        
        # if any keypoints are found, find clusters
        if keyps.size > 0:
            
            db = DBSCAN(eps=self.dbEps, min_samples=self.dbSamples).fit(keyps)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            
            if np.asarray((labels==-1)).all():
                pb = False
                coord = (-1,-1)
            elif np.asarray((labels==0)).any():
                # find the geometric center of the cluster
                class_member_mask = (labels==0)
                xy = keyps[class_member_mask]
                ym = int((xy[:,0].max()+xy[:,0].min())/2)
                xm = int((xy[:,1].max()+xy[:,1].min())/2)
                pb = True
                coord = (ym,xm)
            return pb, coord
            
    def detect_and_extract_patch(self, img):
        patch, center = self._create_patch(img)
        pb, pbPos = self._detect_polar_body_patch(patch)
        return pb, pbPos, patch
    
    def detect_and_extract(self,img, visualize=False):
        cy, cx, aveDist = self.coordOO
        y, x , h, w = self.coordROI
        roi = img[y:y+h, x:x+w]

        alpha = np.linspace(0,18,num=10)*5/180*np.pi
        circy = (np.cos(alpha)*(aveDist+5)).astype(np.uint8)
        circx = (np.sin(alpha)*(aveDist+5)).astype(np.uint8)
        
        coord = np.vstack([ np.hstack([cy-np.flipud(circy[0:5]), cy-circy[1:],
                                       cy-np.flipud(circy[0:-1]*(-1)), cy-circy[1:5]*(-1)]),
                            np.hstack([np.flipud(circx[0:5])+cx, circx[1:]*(-1)+cx,
                                      np.flipud(circx[0:-1]*(-1))+cx, circx[1:5]+cx])])
        coord[coord<20] = 20
        pbpos = np.zeros([2,1], dtype=np.uint64)
        mask = np.zeros((40,40), dtype=bool)
        mask[0:6,0:6] = True
        mask[-5:,0:6] = True
        mask[0:6,-5:] = True
        mask[-5:, -5:] = True
        
        if visualize:
            roiTemp = roi.copy()
            for y,x in zip(coord[0,:], coord[1,:]):
                for i in range(-1,2):
                    roiTemp[draw.polygon_perimeter([y-20+i, y-20+i, y+20-i, y+20-i, y-20-i],
                                                   [x-20+i, x+20-i, x+20-i, x-20+i, x-20+i],
                                                   shape=roiTemp.shape)] = 1
        
        for i in range(0,coord.shape[1]):
            patch = roi[coord[0,i]-20:coord[0,i]+20, coord[1,i]-20:coord[1,i]+20].copy()
            patchKT = feature.corner_shi_tomasi(patch, sigma=2)
            patchKT = (patchKT-patchKT.min())/(patchKT.max()-patchKT.min())
            patchTH = patchKT.copy()
            th = 0.1
            patchTH[patchKT<=th] = 0
            patchTH[patchKT>th] = 1
            patchTH[mask] = 0
            
            keyps = np.where(patchTH==1)
            keyps = np.asarray([[y, x] for (y,x) in zip(keyps[0], keyps[1])])
    
            if keyps.size > 0:
                
                db = DBSCAN(eps=self.dbEps, min_samples=self.dbSamples).fit(keyps)
                labels = db.labels_
                if np.asarray((labels==0)).any():
                    class_member_mask = (labels==0)
                    xy = keyps[class_member_mask]
                    pbpos = np.hstack([pbpos,(np.vstack([coord[0,i]+xy[:,0]-20,coord[1,i]+xy[:,1]-20]))])
        
        pbpos = np.transpose(pbpos).astype(np.uint64)
        pbCoord = pd.DataFrame(data=pbpos, columns={'y','x'})
        pbCoord.drop_duplicates(['x','y'], inplace=True)
        
        db = DBSCAN(eps=25, min_samples=700).fit(pbCoord.values)
        labels = db.labels_
        
        if np.asarray((labels==-1)).all():
            pb = False
            pbPos = (-1,-1)
            if visualize:
                roiPB = roi.copy()
        elif np.asarray((labels==0)).any():
            # find the geometric center of the cluster
            pb = True
            class_member_mask = (labels==0)
            xy = pbCoord.values[class_member_mask]
            pby = int((xy[:,0].max()+xy[:,0].min())/2)
            pbx = int((xy[:,1].max()+xy[:,1].min())/2)
            pbPos = (pby, pbx)
            if visualize:
                roiPB = roi.copy()
                roiPB[xy[:,0], xy[:,1]] = 1

        inPosition = False
        
        if visualize:
            return pb, pbPos, inPosition, roiTemp, roiPB, pbCoord
        else:
            return pb, pbPos, inPosition