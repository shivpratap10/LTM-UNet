# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:54:59 2026

@author: Santosh Prakash
"""

import albumentations as A

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])