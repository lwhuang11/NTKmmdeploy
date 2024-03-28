import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class PersonKeypoints:
    def __init__(self):
        self.keypoints_30_list = []
        self.keypoints_5window = []  
        self.person_count = {}
        self.windowsize = 30
        self.overlap = 10
        self.prenumper = 0
        
        self.model = load_model('LSTM_nor_1000_best1.h5')
        
    def update_keypoints(self, keypoints):
        
        keypoint_person = keypoints['keypoints'] # 取出keypoints資訊
        
        if self.prenumper != len(keypoint_person): # 人數有變動
            # 全部資料刷新
            self.keypoints_30_list = []
            self.person_count = {}
            
        self.prenumper = len(keypoint_person) # 紀錄當前人數
        
        if len(keypoint_person) == 1: # 偵測到一個人
            
            if self._is_valid_person(keypoint_person[0]): # 檢查是否26個點都有抓到
                if len(self.keypoints_30_list) == 0: # 剛開始或數據被洗過
                    
                    self.keypoints_30_list.append(keypoint_person[0]) 
                    self.person_count['person_0'] = 1 # 一筆資料
                    
                else: # 已經有骨架點資料
                    self.keypoints_30_list.append(keypoint_person[0]) 
                    self.person_count['person_0'] += 1
                
            else: # 有點沒抓到
                # 刷新
                self.keypoints_30_list = []  
                self.person_count = {}
        elif len(keypoint_person) > 1: # 偵測到多人
            
            for idx, keyp_eachper in enumerate(keypoint_person):
                if self._is_valid_person(keyp_eachper):
                    if len(self.keypoints_30_list) == 0:
                        for i in range(len(keypoint_person)):
                            name = f"person_{i}"
                            self.keypoints_30_list.append([])
                            self.person_count[name] = 0
                        self.keypoints_30_list[idx].append(keyp_eachper)
                        self.person_count['person_0'] = 1
                    else:
                        name = f"person_{idx}"
                        self.keypoints_30_list[idx].append(keyp_eachper)
                        self.person_count[name] += 1
                else:
                    self.keypoints_30_list = []  
                    self.person_count = {}
                        
        self.check_keypoints()
        self.fight_detect()

    def _is_valid_person(self, person_keypoints):
        
        return len(person_keypoints) == 26 and all(len(coordinate) == 2 for coordinate in person_keypoints)

    def detect(self):
        if len(self.person_count) == 1:
            sequence = np.zeros((30, 26, 2))
            for i in range(30):
                for j in range(26):
                    sequence[i][j][0] = self.keypoints_30_list[i][j][0]
                    sequence[i][j][1] = self.keypoints_30_list[i][j][1]
            
            sequence = sequence.reshape(-1)
            scaler = MinMaxScaler()
            sequence_normalized_flat = scaler.fit_transform(sequence)
            sequence_normalized = sequence_normalized_flat.reshape(1, 30, 52)
            pre = self.model(sequence_normalized)
            labels = (pre >= 0.5).astype(int)
            self.keypoints_5window[0] = labels
            self.keypoints_30_list = self.keypoints_30_list[-20:]
            
        else:
            for idx, keypoints in enumerate(self.keypoints_30_list):
                if len(self.keypoints_30_list[idx]) == 30:
                    sequence = np.zeros((30, 26, 2))
                    for i in range(30):
                        for j in range(26):
                            sequence[i][j][0] = self.keypoints_30_list[idx][i][j][0]
                            sequence[i][j][1] = self.keypoints_30_list[idx][i][j][1]
                    
                    sequence = sequence.reshape(-1)
                    scaler = MinMaxScaler()
                    sequence_normalized_flat = scaler.fit_transform(sequence)
                    sequence_normalized = sequence_normalized_flat.reshape(1, 30, 52)
                    pre = self.model(sequence_normalized)
                    labels = (pre >= 0.5).astype(int)
                    self.keypoints_5window[idx] = labels
                    self.keypoints_30_list[idx] = self.keypoints_30_list[idx][-20:] # 從後面開始20個
        
    def check_keypoints(self):
        if len(self.person_count) == 1:
            if len(self.keypoints_30_list) == 30:
                self.keypoints_5window.append([])
                answer = self.detect()
        elif len(self.person_count) > 1:
            if [len(person_kp) for person_kp in self.keypoints_30_list] == 30:
                [self.keypoints_5window.append([]) for i in range(len(self.person_count))]
                answer = self.detect()
                
    def fight_detect(self):
        for idx, conti_labels in enumerate(self.keypoints_5window):
            if len(conti_labels) == 5:
                if sum(conti_labels) >= 3:
                    print('Fight')
                    self.keypoints_5window[idx] = self.keypoints_5window[idx][1:5] # 保留後五個
            
        