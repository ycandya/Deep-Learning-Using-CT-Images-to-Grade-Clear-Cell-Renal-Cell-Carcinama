import torch.utils.data as data
import random
import torch
from PIL import Image
import os


class Timing_CCR(data.Dataset): 
    def __init__(self, root, transform, train=True): 
        
        self.root = root
        self.transform = transform
        self.train_labels = []
        self.test_labels = []
        self.patient = []  
        self.org_labels = []
        self.train = train  
        SE1_path = '/home/dingzicha/Medical/ccrcc/SE1_20211012/'
        
        if self.train == True:
            self.train_imgs=[]
            with open('%s/time_new_train_info.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = entry[0].split('/')[-1]
                    self.train_labels.append(int(entry[2]))   
                    # self.train_imgs.append(img_path) 
                    self.train_imgs.append(os.path.join(SE1_path, img_path)) 
                    # self.train_imgs.append(entry[0]) 
                    self.patient.append(float(entry[1].replace('-', '.')))
                    self.org_labels.append(int(entry[0].split('/')[-1][0]))
        else:
            self.test_imgs=[]
            with open('%s/time_new_test_info.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = entry[0].split('/')[-1]
                    self.test_labels.append(int(entry[2]))   
                    # self.train_imgs.append(img_path) 
                    self.test_imgs.append(os.path.join(SE1_path, img_path)) 
                    # self.test_imgs.append(entry[0])
                    self.patient.append(float(entry[1].replace('-', '.')))
                    self.org_labels.append(int(entry[0].split('/')[-1][0]))
          
    def __getitem__(self, index):  
        if self.train:
            img_path = self.train_imgs[index]
            target = self.train_labels[index] 
            patient = self.patient[index]   
            org_label = self.org_labels[index] 
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, patient, img_path, org_label
        else:
            img_path = self.test_imgs[index]
            target = self.test_labels[index] 
            patient = self.patient[index]  
            org_label = self.org_labels[index] 
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)  
            return img, target, patient, img_path, org_label
     
    def __len__(self):
        if self.train:
            return len(self.train_imgs) 
        else:
            return len(self.test_imgs)


class self_supCCR(data.Dataset): 
    def __init__(self, root, transform, num_class=4, train=True): 
        
        self.root = root
        self.transform = transform
        self.train_labels = []
        self.test_labels = []
        self.patient = []  
        self.org_labels = []
        self.train = train  
        SE1_path = 'xxxxxxxxxxx'        
        
        if self.train == True:
            self.train_imgs=[]
            with open('%s/xxxxxxxxxxx.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = entry[0].split('/')[-1]
                    self.train_labels.append(int(entry[2]))   
                    self.train_imgs.append(os.path.join(SE1_path, img_path)) 
                    self.patient.append(float(entry[1].replace('-', '.')))
                    self.org_labels.append(int(entry[0].split('/')[-1][0]))
        else:
            self.test_imgs=[]
            with open('%s/xxxxxxxxxxx.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = entry[0].split('/')[-1]
                    self.test_labels.append(int(entry[2]))   
                    self.test_imgs.append(os.path.join(SE1_path, img_path))  
                    self.patient.append(float(entry[1].replace('-', '.')))
                    self.org_labels.append(int(entry[0].split('/')[-1][0]))
          
    def __getitem__(self, index):  
        if self.train:
            # rotate the CT image, the labels were changed into 0, 1, 2, 3
            k = random.randint(0,3)
            img_path = self.train_imgs[index]
            target = k
            patient = self.patient[index]    
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            # 
            image = torch.rot90(image, k, [1,2])
            return image, target, patient
        else:
            k = random.randint(0,3)
            img_path = self.test_imgs[index]
            target = k
            patient = self.patient[index]  
            image = Image.open(img_path).convert('RGB')   
            image = self.transform(image)
            image = torch.rot90(image, k, [1,2])
            return image, target, patient
     
    def __len__(self):
        if self.train:
            return len(self.train_imgs) 
        else:
            return len(self.test_imgs)


if __name__ == '__main__':
    pass