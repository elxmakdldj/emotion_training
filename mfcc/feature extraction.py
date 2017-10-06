import librosa
import csv

output=1
with open('./mfcc.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(1, 201): #1~50:기쁨, 51~100:슬픔, 101~150:평정, 151~200:화남
        filename = 'C:\\Users\\Newyear\\Desktop\\R&D\\%d.wav' %i
        y, sr = librosa.load(filename)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) #mfcc 추출값은 20차원의 벡터
        a = []
         #mfcc 추출 평균값
        s = 0
        for j in mfcc:
            for k in j:
                s=s+k
            s=s/len(j)
            a.append(s)
        a.append(output)    # output-1:기쁨 2:슬픔 3:평정 4:화남
        if i%50==0:
            output=output+1
        writer.writerow(a)
