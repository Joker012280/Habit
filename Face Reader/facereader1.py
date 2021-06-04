# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:45:32 2018

@author: BME
"""
import sys
import torch
import torch.nn.init
from torch.autograd import Variable
import torchvision.utils as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import time
import zipfile
import random

import cv2  # 얼굴 인식을 위한 opencv 설치
import numpy
from matplotlib import pyplot as plt

output_dir = "output"
usr_name = "DH"

start =time.time()
#filename = raw_input()
filename = sys.argv[1]
filename = './picture/1.jpg'
cascadefile = "./haarcascade_lefteye_2splits.xml"  # 왼쪽 눈을 인식
cascadefile1 = "./haarcascade_righteye_2splits.xml" # 오른쪽 눈을 따로 인식
img = cv2.imread(filename) 
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 컬러 이미지를 인식할 수 있는 흑백으로 바꿔 줌


# 왼쪽 눈에 대해 왼쪽 눈 부분만 추출
cascade = cv2.CascadeClassifier(cascadefile)
facelist = cascade.detectMultiScale(imgray, scaleFactor=2.08, minNeighbors=1) 


cropped = []
if len(facelist) >= 1: 
    for face in facelist: 
        x, y, w, h = face 
        cv2.rectangle(imgray, (x, y), (x+w, y+h), (255, 0, 0), 2) # 눈에 해당하는 부위를 네모로 표시
        cropped = imgray[y:y+h, x:x+w]  # 눈에 해당하는 부분을 추출함
    result_filename = ["./real/left/1.jpg"]
    result_filename = ''.join(result_filename)
    cv2.imwrite(result_filename,cropped)   # 추출한 눈 부위를 저장
if not np.any(cropped):     # 눈을 인식하지 못했을 때
    print('왼쪽 눈을 인식하지 못했습니다..ㅜㅠ')  

# 오른쪽 눈에 대해 오른쪽 눈 부분만 추출
cascade = cv2.CascadeClassifier(cascadefile1)
facelist = cascade.detectMultiScale(imgray, scaleFactor=2.08, minNeighbors=1) 

cropped=[]
if len(facelist) >= 1: 
    for face in facelist: 
        x, y, w, h = face 
        cv2.rectangle(imgray, (x, y), (x+w, y+h), (255, 0, 0), 2)   # 눈에 해당하는 부위를 네모로 표시
        cropped = imgray[y:y+h, x:x+w]   # 눈에 해당하는 부분을 추출함
    result_filename = ["./real1/right/1.jpg"]   
    result_filename = ''.join(result_filename)
    cv2.imwrite(result_filename,cropped)  # 추출한 눈 부위를 저장
if not np.any(cropped):    # 눈을 인식하지 못했을 때
    print('오른쪽 눈을 인식하지 못했습니다..ㅠㅜ')
 
    
transform = transforms.Compose(
    [transforms.Resize(24),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size=16
testset = torchvision.datasets.ImageFolder(root='./real',transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=testset, 
                                          batch_size = batch_size,
                                          shuffle=True)  # 추출했던 왼쪽눈을 testset으로 설정

testset1 = torchvision.datasets.ImageFolder(root='./real1',transform=transform)
test_loader1 = torch.utils.data.DataLoader(dataset=testset1, 
                                          batch_size = batch_size,
                                          shuffle=True)  # 추출했던 오른쪽 눈을 testset으로 설정

test_images_l, test_labels_l = next(iter(test_loader))
test_images_r, test_labels_r = next(iter(test_loader1))


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(3,48,3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(48),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(48,96,3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(96*4*4,120),
            torch.nn.ReLU(),            
            torch.nn.Linear(120,84),
            torch.nn.ReLU(),
            torch.nn.Linear(84,3)
        )
        
    def forward(self,x):
        x=self.layer1(x)
        x=x.view(x.size()[0],-1)
        x=self.fc(x)
        return x

left_eye = LeNet()   # 모델 클래스 입력
left_eye.load_state_dict(torch.load('./Left_eye.path'))
left_eye.eval()

right_eye = LeNet()
right_eye.load_state_dict(torch.load('./right_eye.path'))
right_eye.eval()

size_eye = LeNet()
size_eye.load_state_dict(torch.load('./size_eye.path'))
size_eye.eval()

X = Variable(test_images_l.view(-1,3,24,24).float())
E1 = left_eye(X)   # 왼쪽 눈에 대해 up, middle, down 판정
E3 = size_eye(X)  # 눈의 크기를 판정

Y = Variable(test_images_r.view(-1,3,24,24).float())
E2 = right_eye(Y)   # 오른 쪽 눈에 대해 up, middle, down 판정
E4 = size_eye(Y)  # 눈의 크기를 판정


classes1 = ('down','middle','up')
classes2 = ('down','middle','up')
classes3 = ('big', 'small')


# 눈 크기와 눈꼬리에 대한 클래스 출력
print('왼쪽 눈꼬리는 '+classes1[torch.max(E1,1)[1][0]]+' 되어 있습니다!')
print('오른쪽 눈꼬리는 '+classes2[torch.max(E2,1)[1][0]]+' 되어 있습니다!')
print('왼쪽 눈의 크기는 '+classes3[torch.max(E3,1)[1][0]]+' 하군요!')
print('오른쪽 눈의 크기는 '+classes3[torch.max(E4,1)[1][0]]+' 하군요!')



# 눈이 클 때

L1  = "감정표현이 뛰어나다."
L2 = "감정 중시하며, 천진하고, 착하다."
L3 = "동정심 많음, 금전이나 애정 문제로 남에게 쉽게 이용당할 수 있다."
L4 = "애정에서는 우유부단하고 주저하여 결정을 내리지 못하는 경우가 있고, 심지어 양다리를 걸치는 상황도 생길 수 있다."
L5 = "시야가 넓고 명랑하고 외향적이며 사교와 단체 생활을 좋아한다."
L6 = "관찰력이 예리하고 반응이 민첩하다."
L7 = "색채 분별력이 뛰어나고 음악이나 회화 쪽으로 재능을 발휘할 수 있다."
L8 = "목표를 이루기 위한 의지와 집중력이 부족하기 때문에 전문 분야로 성과를 거두기 어려울 수 있다."
L9 = "언변이 좋아 이성의 환심을 살 수 있다."
L10 = "마음이 열려 있어 정이 많고, 열정적이다."
L11 = "호기심이 넘치고 개방적인 성격을 갖추고 있다."
L12 = "정이 많아 이성에 대한 관심과 인기도도 많고 개방적인 성격을 갖고 있다."
L13 = "적극적인 애정공세를 펴는 경우가 많다."
L14 = "심리 변화가 심하기 때문에 즉흥적인 행동을 보여 오해를 받는 경우가 많다."
L15 = "현실보다는 이상을 추구하여 금전적으로 기복이 심하다."
L16 = "일반적으로 얼굴을 보았을 때 크다고 느껴지는 눈을 가진 사람은 감각이 뛰어나고 이성을 끌어들이는 매력이 있으며 개방적이다."
L17 = "정열적인 성격을 갖추고 있으며 상대방을 잘 배려해주는 한편, 상대방의 마음을 읽어내는 재능이 있다."
L18 = "개방적인 성격이기는 하지만, 사람을 가려서 사귀는 편이고 정열이 지나치게 강해서 애정문제에 빠지면 헤어나지 못한다."
L19 = "사랑을 할 때에는 최선을 다 하지만, 사랑이 식으면 미련 없이 등을 돌리는 냉정함이 있다."
L20 = "남성의 경우에는 리더가 될 수 있는 자질을 충분히 갖추고 있기 때문에 다른 사람 밑에서 일하는 것에 거부감을 느낀다. 단, 직장생활을 하면 승진이 빠른 편이다."
L21 = "여성의 경우에는 남성에게 인기가 좋으며 음악적 감각이 뛰어나서 노래를 잘하며 춤에도 소질이 있다."

L = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15, L16, L17, L18, L19, L20, L21]

# 눈이 작을 때

S1 = "차분하고 겸손한 성격을 갖추고 있다."
S2 = "강인하고 냉정한 자기만의 세계를 가진 사람이 많다."
S3 = "말보다는 행동으로 생각을 표현하는 신중함을 가진다."
S4 = "자신의 속내를 쉽게 드러내지 않는다."
S5 = "사회적으로 믿음직하다는 평가를 받는다."
S6 = "한번 마음먹은 일은 가능하면 끝까지 성사시키려는 끈기도 있다."
S7 = "힘든 시기가 닥치더라도 꿋꿋이 이겨낼 수 있는 사람이다."
S8 = "젊은 시절에 고생이 많고 매력이 뒤떨어져 윗사람들의 사랑을 받지 못한다."
S9 = "겸손한 성격으로 대인관계에서 자신을 굽힐 줄 알고 지적인 능력이 뛰어나기 때문에 학문적인 분야에서 성공할 가능성이 높다."
S10 = "특히 한 우물을 파서 성공을 거두는 예가 많지만, 성격이 매우 강해 냉정하다는 인상을 주기 쉽고 자신만의 공간에 틀어박혀 좀처럼 마음을 열지 않는다."
S11 = "남성의 경우 여자를 다루는 능력과 금전을 융통하는 능력은 부족하지만, 믿음직하고 성실하기 때문에 늦게 인정을 받는 타입이다."
S12 = "의지가 강하기 때문에 난관을 잘 극복한다."
S13 = "여성의 경우에는 남성을 선택하는 데 많은 시간이 걸리지만, 한번 마음을 주면 어지간해서는 다른 이성에게 눈길을 돌리지 않는 일편단심형이며 가족을 매우 중요하게 생각한다."
S14 = "가정 경제를 꾸려나가는 능력이 있고 사회활동을 해도 성공할 수 있다."

S = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14]

# 눈꼬리가 올라간 눈
U1 = "성급하며 양의 특성 상 기질이 강하고 빠르고 폭발적이고 급한 것이다."
U2 = "감각이 뛰어나고 어떤 일에도 굽히지 않는 강한 용기를 갖추고 있으며 두뇌회전이 빠르고 기회를 잡는 능력이 뛰어나다."
U3 = "예술적인 방면에 소질이 있고 추친력을 갖추고 있으며 아무리 어려운 난관에 부딪혀도 강한 인내력으로 돌파할 수 있는 용기가 있다."
U4 = "기회가 오면 어떻게 해서든 움켜쥐려하기 때문에 이기적이라는 인상을 주기 쉽고 독단적인 성향이 강하다."
U5 = "남성의 경우에는 두뇌회전이 빨라 중간관리직으로 잘 어울리며 실행력이 있어 운세가 좋은 편이다."
U6 = "자신의 주장을 약간 억제하고 다른 사람의 의견을 받아들이는 포용력을 갖추는 것이 바람직하다."
U7 = "성공을 추구하는 눈이다."
U8 = "성격이 예민하고 반응이 빠르고 결단력이 있고 시기를 놓치지 않는다."
U9 = "그러나 자존심과 승부욕 소유욕이 강하고 의심이 많은 것이 단점이다."
U10 = "품격이 있다."
U11 = "두뇌 회전이 빠르고 총명하다."
U12 = "예상 밖의 아이디어를 가져 영리해 보일 수 있다."
U13 = "남의 어려움을 앞장 서 해결하므로 인복이 많다."
U14 = "애정 문제에서 주도권을 잡고 적극적으로 어필한다."
U15 = "점유욕과 지배욕이 있다."
U16 = "끈기가 있고 체력이 강하다."
U17 = "주관이 분명하고 대범한 성격을 갖췄다."
U18 = "어떤 일을 하든 반드시 성사시키는 강인함을 갖추고 있다."
U19 = "리더십도 매우 뛰어나 ‘대인의 상’, ‘장수의 상’이라고 표현한다."
U20 = "자존심도 강해 다른 사람에게 지는 것을 싫어한다."
U21 = "자신의 영역을 침범당하면 즉각 자기방어에 나설 정도로 철저한 자기관리 능력을 자랑한다."

U = [U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21]

# 눈꼬리가 내려간 눈

O1 = "만약 반대로 눈끝이 아래로 숙인 자는 음에 속하니 문질이며 부드럽고 약하며 침착하며 느린 것이다."
O2 = "눈꼬리가 처진 눈을 가진 사람은 심리적으로 느긋하고 여유 있는 성격이며 투쟁이나 다툼보다는 평화를 사랑한다."
O3 = "모든 일을 긍정적이고 원만하게 처리하려 하기 때문에 대인관계가 매우 좋아 다른 사람의 도움으로 출세를 할 가능성이 매우 높다. "
O4 = "성실하다는 점도 장점이다."
O5 = "수동적이며 소극적이기 때문에 주위 사람들로부터 자신의 주장을 할 줄 모르는 사람이라는 비난을 받는다."
O6 = "남성의 경우에는 친구나 동료, 선후배와의 관계가 원만해서 일찍 출세할 수 있다."
O7 = "여성의 유혹에 넘어가기 쉽고 그 때문에 실패를 맛볼 가능성이 높다."
O8 = "여성의 경우에는 역시 남성의 유혹에 넘어가기 쉽고 그 때문에 손해를 볼 가능성이 높다."
O9 = "사교적이며 인정이 많다."
O10 = "인정이 많고 보스 기질이 있다."
O11 = "대인관계가 좋고 주변에 사람이 많이 모이는 편이다."
O12 = "유머도 풍부하여 재미있고 즐거운 인생을 보낼 것 같으나 사실 외로움도 많이 탄다."
O13 = "이성에 대한 호기심이 매우 강해서 정 때문에 마음을 졸일 가능성이 매우 높다."
O14 = "모든 사람에게 친절하고 다정하게 행동하지만, 그에 못지 않을 정도로 자존심이 강하고 보스 기질이 강하기 때문에 실질적으로는 성격이 매우 강한 사람이다."

O = [O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14]

# 눈꼬리가 일직선으로 수평인 눈
M1 = "불상불하의 눈으로 불투(사물을 훔쳐보지 말아야)해야 모름지기 쓸 만한 그릇이 된다."
M2 = "위인(爲人)이 강개롭고 심평정직하다."
M3 = "위, 아래로 향하지 않고 수평을 유지하는 것이 가장 이상적이다."

M = [M1, M2, M3]

# 짝짝이 눈

D1 = "성격이 변덕스럽고 우유부단하다."
D2 = "부모님의 사이가 좋지않을 확률이 높다."
D3 = "성격상 소극적이면서 어두운 면이 있다."
D4 = "남다른 관찰력과 예민한 직감력을 지녔다."
D5 = "인생 굴곡이 많다."
D6 = " 활동적이고 야심이 있고 부를 축적한다."
D7 = "세상을 두가지 관점으로 보는 경향이 있어 객관성이 매우 뛰어나고 논리적이다."
D8 = "어떤 분야에서든 상위에까지 오르기는 하지만, 최상위에 오르기는 어려움이 있다."
D9 = "주변에 시기와 질투를 하는 사람들이 많다."
D10 = "한쪽은 크고 한쪽은 작은 눈을 가진 사람은 인생에서 큰 전환기를 겪을 가능성이 높고 두뇌회전이 빠른 편이다."
D11 = "자기 주장이 뚜렷하고 활동적이며 승부에 대한 열정이 강하고 이상도 높다."
D12 = "고집이 세고 자기 주장이 강해 견제의 대상이 될 가능성이 높고 이성에게 약한 편이며 인생에 기복이 아주 심하다."
D13 = "남성의 경우에는 왼쪽이 클 경우에는 매우 활동적이고 승부욕이 강하며 이상이 높고, 오른쪽이 클 경우에는 정에 이끌리기는 해도 리더심과 자신감이 있어서 노력에 따라 행복을 만끽할 수 있다."

D = [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13]


# 눈꼬리
n=4
if classes1[torch.max(E1,1)[1][0]] == classes2[torch.max(E2,1)[1][0]]:  # 양쪽 눈꼬리가 같을 때
    n=4
    if classes1[torch.max(E1,1)[1][0]] == 'down':  # 둘다 눈꼬리가 내려갔으면
        ind = random.sample(range(14),n)
        answer=[]
        for i in ind:
            answer.append(O[i])
    elif classes1[torch.max(E1,1)[1][0]] == 'up':   # 둘다 눈꼬리가 올라갔으면
        ind = random.sample(range(21),4)
        answer=[]
        for i in ind:
            answer.append(U[i])
    else:    # 둘다 눈꼬리가 수평에 이르면
        answer=[]
        for i in range(3):
            answer.append(M[i])
elif classes1[torch.max(E1,1)[1][0]]  or classes1[torch.max(E2,1)[1][0]] == 'middle':
    n=3
    if classes1[torch.max(E1,1)[1][0]] or classes1[torch.max(E2,1)[1][0]] == 'up':  # 눈꼬리가 수평과 올라갔다면
        ind = random.sample(range(21),n)
        answer=[]
        for i in ind:
            answer.append(U[i])
    else:      # 눈꼬리가 수평과 내려갔다면
        ind = random.sample(range(14),n)
        answer=[]
        for i in ind:
            answer.append(O[i])
else:
    n=2
    ind = random.sample(range(14),n)
    answer=[]
    for i in ind:
            answer.append(O[i])
            answer.append(U[i])
    
# 눈 크기
n=4
if classes3[torch.max(E3,1)[1][0]] == classes3[torch.max(E4,1)[1][0]]:   # 양쪽 눈의 크기가 같으면
    if classes3[torch.max(E3,1)[1][0]] == 'big' :   # 눈의 크기가 클 때
        ind = random.sample(range(21),n)
        for i in ind:
            answer.append(L[i])
    else:
        ind = random.sample(range(14),n)
        for i in ind:
            answer.append(S[i])
else:   # 두 눈의 결과가 다를 때
    ind = random.sample(range(14),n)
    answer.append(L[ind[0]])
    answer.append(S[ind[1]])
    answer.append(D[ind[2]])
    answer.append(D[ind[3]])
    
print('수행 시간은 '+str(time.time()-start)+ '초 걸렸습니다.')

with open(output_dir+"+.txt", "w") as f:
    f.write('< ')
    f.write(usr_name)
    f.write(' 님의 관상 결과!!!! > \n\n')
    f.write('\n'.join(answer))
        
